import torch
import time

from pathlib import Path
from config.parser import Config

from models.dynamics import Dynamics
from models.policy import Policy


def trace_model(dynamics: Dynamics, policy: Policy):
    origin_device = dynamics.device

    dynamics.eval()
    policy.eval()

    dynamics_inputs_example = torch.randn(1, dynamics.input_dim).to(torch.device("cpu"))
    dynamics_h_example = torch.randn(1, dynamics.rnn_depth, dynamics.hidden_dim).to(torch.device("cpu"))
    dynamics_traced = torch.jit.trace(
        dynamics.to(torch.device("cpu")), example_inputs=(dynamics_inputs_example, dynamics_h_example))

    policy_inputs_example = torch.randn(1, policy.input_dim).to(torch.device("cpu"))
    policy_h_example = torch.randn(1, dynamics.rnn_depth, dynamics.hidden_dim).to(torch.device("cpu"))
    policy_traced = torch.jit.trace(
        policy.to(torch.device("cpu")), example_inputs=(policy_inputs_example, policy_h_example))
    dynamics_traced = torch.jit.optimize_for_inference(dynamics_traced)
    policy_traced = torch.jit.optimize_for_inference(policy_traced)

    dynamics.to(origin_device)
    policy.to(origin_device)

    for name, param in dynamics_traced.named_parameters():
        if param.device != torch.device("cpu"):
            print(name, param.device)
    for name, buf in dynamics_traced.named_buffers():
        if buf.device != torch.device("cpu"):
            print(name, buf.device)
    
    for name, param in policy_traced.named_parameters():
        if param.device != torch.device("cpu"):
            print(name, param.device)
    for name, buf in policy_traced.named_buffers():
        if buf.device != torch.device("cpu"):
            print(name, buf.device)
    
    return dynamics_traced.to(torch.device("cpu")), policy_traced.to(torch.device("cpu"))   


def export_model(dynamics_only: bool = False):
    config_path = Path(__file__).resolve().parent.parent.parent / "config"
    config_model = Config.load(config_path / "models_config.yaml")
    config_training = Config.load(config_path / "training_config.yaml")

    script_path = Path(__file__).resolve()
    train_root = script_path.parent.parent
    save_folder = train_root / "checkpoints" / config_training.policy.get("name")

    dynamics_name = config_training.policy.get("dynamics_name")
    dynamics_pth = save_folder.parent / dynamics_name / \
        f"dynamics_{config_training.policy.get('dynamics_load_type')}.pth"
    dynamics = Dynamics(config_model, torch.device("cpu"), dynamics_pth)
    print(f"Dynamics model total parameters: {dynamics.total_params}")
    dynamics.eval()

    dynamics_inputs_example = torch.randn(1, dynamics.input_dim).to(torch.device("cpu"))
    dynamics_h_example = torch.randn(1, dynamics.rnn_depth, dynamics.hidden_dim).to(torch.device("cpu"))
    dynamics_traced = torch.jit.trace(dynamics, example_inputs=(dynamics_inputs_example, dynamics_h_example))
    # dynamics_traced = torch.jit.freeze(dynamics_traced)
    dynamics_traced = torch.jit.optimize_for_inference(dynamics_traced)

    if not dynamics_only:
        load_name = config_training.policy.get("load_name")
        if load_name is not None:
            load_path = save_folder.parent / load_name / \
                f"policy_{config_training.policy.get('policy_load_type')}.pth"
        policy = Policy(config_model, torch.device("cpu"), load_path)
        print(f"Policy model total parameters: {policy.total_params}")
        policy.eval()

        policy_inputs_example = torch.randn(1, policy.input_dim).to(torch.device("cpu"))
        if policy.net_type == "mlp":
            policy_h_example = torch.randn(
                1, config_model.policy.net_params.mlp.history_window, policy.obs_dim + policy.execute_ctl_dim).to(torch.device("cpu"))
        else:
            policy_h_example = torch.randn(1, dynamics.rnn_depth, dynamics.hidden_dim).to(torch.device("cpu"))
        policy_traced = torch.jit.trace(policy, example_inputs=(policy_inputs_example, policy_h_example))
        # policy_traced = torch.jit.freeze(policy_traced)
        policy_traced = torch.jit.optimize_for_inference(policy_traced)
    
    with torch.no_grad():
        start = time.time()
        y_ts = dynamics_traced(dynamics_inputs_example, dynamics_h_example)
        print(f"Dynamics traced inference time: {time.time() - start} seconds")
        start = time.time()
        y_py = dynamics(dynamics_inputs_example, dynamics_h_example)
        print(f"Dynamics python inference time: {time.time() - start} seconds")
        torch.testing.assert_close(y_py, y_ts, rtol=1e-5, atol=1e-6)

        if not dynamics_only:
            start = time.time()
            y_ts = policy_traced(policy_inputs_example, policy_h_example)
            print(f"Policy traced inference time: {time.time() - start} seconds")
            start = time.time()
            y_py = policy(policy_inputs_example, policy_h_example)
            print(f"Policy python inference time: {time.time() - start} seconds")
            torch.testing.assert_close(y_py, y_ts, rtol=1e-5, atol=1e-6)

    dynamics_traced.save(save_folder / "dynamics_traced.pt")
    if not dynamics_only:
        policy_traced.save(save_folder / "policy_traced.pt")
    print(f"Exported models saved to {save_folder}")

    with torch.no_grad():
        dynamics = dynamics.to(torch.device("cuda"))
        dynamics_inputs_example = dynamics_inputs_example.to(torch.device("cuda"))
        dynamics_h_example = dynamics_h_example.to(torch.device("cuda"))
        dynamics_traced = torch.jit.trace(dynamics, example_inputs=(dynamics_inputs_example, dynamics_h_example))
        dynamics_traced = torch.jit.optimize_for_inference(dynamics_traced)

        start = time.time()
        y_ts = dynamics_traced(dynamics_inputs_example, dynamics_h_example)
        print(f"(cuda) Dynamics traced inference time: {time.time() - start} seconds")
        start = time.time()
        y_py = dynamics(dynamics_inputs_example, dynamics_h_example)
        print(f"(cuda) Dynamics python inference time: {time.time() - start} seconds")
        torch.testing.assert_close(y_py, y_ts, rtol=1e-5, atol=1e-6)

        if not dynamics_only:
            policy = policy.to(torch.device("cuda"))
            policy_inputs_example = policy_inputs_example.to(torch.device("cuda"))
            policy_h_example = policy_h_example.to(torch.device("cuda"))
            policy_traced = torch.jit.trace(policy, example_inputs=(policy_inputs_example, policy_h_example))
            policy_traced = torch.jit.optimize_for_inference(policy_traced)

            start = time.time()
            y_ts = policy_traced(policy_inputs_example, policy_h_example)
            print(f"(cuda) Policy traced inference time: {time.time() - start} seconds")
            start = time.time()
            y_py = policy(policy_inputs_example, policy_h_example)
            print(f"(cuda) Policy python inference time: {time.time() - start} seconds")
            torch.testing.assert_close(y_py, y_ts, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    export_model(dynamics_only=False)
