import unittest
import torch
import numpy as np
from models.stu.model import SSSM, SSSMConfigs
from torch.nn import MSELoss
from safetensors.torch import load_file
from torch.optim import AdamW


class TestModelPredictions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {cls.device}")

        cls.configs = SSSMConfigs(
            n_layers=2,
            n_embd=512,
            d_out=512,
            sl=300,
            scale=4,
            bias=False,
            dropout=0.10,
            num_eigh=24,
            auto_reg_k_u=3,
            auto_reg_k_y=2,
            learnable_m_y=True,
            loss_fn=MSELoss(),
            controls={"task": "mujoco-v3", "controller": "Ant-v1"},
        )

        cls.model = SSSM(cls.configs).to(cls.device)
        cls.model = torch.compile(cls.model)
        # model_path = "_sssm-Ant-v1-model_step-239-2024-06-24-14-23-16.pt"
        model_path = "pls_work.pt"
        state_dict = load_file(model_path, device="cuda:0")
        cls.model.load_state_dict(state_dict)
        cls.model.eval()

        cls.sequence_length = 300
        cls.input_dim = 512
        cls.batch_size = 2

        # Load a subset of the test data
        cls.test_data = torch.load(
            "data/mujoco-v3/Ant-v1/Ant-v1_ResNet-18_test.pt", map_location=cls.device
        )
        cls.subset_size = 10
        cls.test_subset = cls.test_data[: cls.subset_size]

    def generate_random_batch(self):
        return torch.randn(self.batch_size, self.sequence_length, self.input_dim).to(
            self.device
        )

    def test_model_output_shape(self):
        """Test if the model produces output of the expected shape."""
        input_data = self.generate_random_batch()
        with torch.no_grad():
            output, _ = self.model.forward(input_data, input_data)
        self.assertEqual(
            output.shape,
            input_data.shape,
            "Model output shape doesn't match input shape",
        )

    def test_different_random_inputs(self):
        """Test if the model produces different outputs for different random inputs."""
        with torch.no_grad():
            input1 = self.generate_random_batch()
            input2 = self.generate_random_batch()

            pred1, _ = self.model.forward(input1, input1)
            pred2, _ = self.model.forward(input2, input2)

            diff = torch.abs(pred1 - pred2).mean().item()
            print(f"Mean difference between predictions: {diff}")

            self.assertFalse(
                torch.allclose(pred1, pred2, atol=1e-4),
                "Model produced identical predictions for different random inputs",
            )

    def test_same_input_consistency(self):
        """Test if the model produces the same output for the same input."""
        with torch.no_grad():
            input_data = self.generate_random_batch()

            pred1, _ = self.model.forward(input_data, input_data)
            pred2, _ = self.model.forward(input_data, input_data)

            self.assertTrue(
                torch.allclose(pred1, pred2, atol=1e-6),
                "Model produced different predictions for the same input",
            )

    def test_batch_independence(self):
        """Test if predictions for one batch element are independent of other batch elements."""
        with torch.no_grad():
            input_data = self.generate_random_batch()

            full_pred, _ = self.model.forward(input_data, input_data)

            individual_preds = []
            for i in range(self.batch_size):
                single_pred, _ = self.model.forward(
                    input_data[i : i + 1], input_data[i : i + 1]
                )
                individual_preds.append(single_pred)

            for i in range(self.batch_size):
                self.assertTrue(
                    torch.allclose(full_pred[i], individual_preds[i][0], atol=1e-6),
                    f"Prediction for batch element {i} is not independent",
                )

    def test_model_gradient_flow(self):
        """Test if gradients are flowing through the model."""
        input_data = self.generate_random_batch()
        target = self.generate_random_batch()

        self.model.train()
        output, _ = self.model.forward(input_data, target)
        loss = self.configs.loss_fn(output, target)
        loss.backward()

        no_grad_params = [
            name for name, param in self.model.named_parameters()
            if param.requires_grad and (param.grad is None or torch.sum(torch.abs(param.grad)) == 0)
        ]

        if no_grad_params:
            print("Parameters without gradient:", no_grad_params)
        else:
            print("All parameters are receiving gradients.")

        self.assertTrue(len(no_grad_params) == 0, "Some parameters are not receiving gradients")
        self.model.eval()

    def test_perfect_prediction_on_subset(self):
        """Test if the model can be trained to predict a subset of test data perfectly."""
        test_model = SSSM(self.configs).to(self.device)
        test_model = torch.compile(test_model)
        test_model.load_state_dict(self.model.state_dict())
        test_model.train()

        optimizer = AdamW(test_model.parameters(), lr=0.001)
        criterion = MSELoss()

        n_epochs = 1000
        target_loss = 1e-6
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            outputs, _ = test_model.forward(self.test_subset, self.test_subset)
            loss = criterion(outputs, self.test_subset)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.6f}")

            if loss.item() < target_loss:
                print(f"Reached target loss at epoch {epoch+1}")
                break

        test_model.eval()
        with torch.no_grad():
            final_outputs, _ = test_model.forward(self.test_subset, self.test_subset)
            final_loss = criterion(final_outputs, self.test_subset)

        print(f"Final loss on training subset: {final_loss.item():.6f}")

        self.assertLess(
            final_loss.item(),
            target_loss,
            f"Model failed to achieve perfect prediction. Final loss: {final_loss.item():.6f}",
        )

        unseen_data = self.test_data[self.subset_size : self.subset_size * 2]
        with torch.no_grad():
            unseen_outputs, _ = test_model.forward(unseen_data, unseen_data)
            unseen_loss = criterion(unseen_outputs, unseen_data)

        print(f"Loss on unseen data: {unseen_loss.item():.6f}")

        self.assertGreater(
            unseen_loss.item(),
            final_loss.item() * 10,
            "Model did not overfit as expected. Loss on unseen data is not significantly higher.",
        )

    def test_model_parameters(self):
        """Test if model parameters are changing during training."""
        initial_params = [param.clone().detach() for param in self.model.parameters()]

        input_data = self.generate_random_batch()
        target = self.generate_random_batch()

        self.model.train()
        optimizer = Adam(self.model.parameters(), lr=0.001)

        for _ in range(10):
            optimizer.zero_grad()
            output, _ = self.model.forward(input_data, target)
            loss = self.configs.loss_fn(output, target)
            loss.backward()
            optimizer.step()

        params_changed = any(
            not torch.allclose(initial, param.detach(), atol=1e-4)
            for initial, param in zip(initial_params, self.model.parameters())
        )

        self.assertTrue(
            params_changed, "Model parameters did not change during training"
        )
        self.model.eval()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    unittest.main()
