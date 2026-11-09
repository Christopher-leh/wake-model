"""Deine PDE-Formeln (Kovasznay, 2D/3D-RANS, Actuator Disk)."""
import torch

def NS2D(
        self,
        coords_col: torch.Tensor,
    ) -> torch.Tensor:
        
        x = coords_col[:, 0:1].detach().requires_grad_(True)
        y = coords_col[:, 1:2].detach().requires_grad_(True)

        out = self.model(torch.cat([x, y], dim=1))
        u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

        # calculate gradients, scaled by the spatial scales(from normalization)

        u_x_norm, u_y_norm = torch.autograd.grad(
            u, [x, y], grad_outputs=torch.ones_like(u), create_graph=True
        )
        v_x_norm, v_y_norm = torch.autograd.grad(
                v, [x, y], grad_outputs=torch.ones_like(u), create_graph=True
        )
        p_x_norm, p_y_norm = torch.autograd.grad(
            p, [x, y], grad_outputs=torch.ones_like(u), create_graph=True
        )

        u_xx_norm = torch.autograd.grad(
            u_x_norm, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_yy_norm = torch.autograd.grad(
            u_y_norm, y, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        v_xx_norm = torch.autograd.grad(
            v_x_norm, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        v_yy_norm = torch.autograd.grad(
            v_y_norm, y, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]

        u_x = self.x_scale * u_x_norm  # ∂/∂x = 2 / (x_max - x_min) · ∂/∂x_norm
        u_y = self.y_scale * u_y_norm  # ∂/∂y = 2 / (y_max - y_min) · ∂/∂y_norm

        v_x = self.x_scale * v_x_norm
        v_y = self.y_scale * v_y_norm

        p_x = self.x_scale * p_x_norm
        p_y = self.y_scale * p_y_norm

        u_xx = self.x_scale**2 * u_xx_norm
        u_yy = self.y_scale**2 * u_yy_norm

        v_xx = self.x_scale**2 * v_xx_norm
        v_yy = self.y_scale**2 * v_yy_norm

        res_mom_x = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        res_mom_y = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        res_cont = u_x + v_y

        return res_mom_x, res_mom_y, res_cont