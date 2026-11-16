"""Deine PDE-Formeln (Kovasznay, 2D/3D-RANS, Actuator Disk)."""

import torch


def NS2D(
    model: torch.nn.Module,
    coords_col: torch.Tensor,
    pde_terms: torch.Tensor,
    scales: tuple[float, float],
    nu: float,
) -> torch.Tensor:

    x = coords_col[:, 0:1].detach().requires_grad_(True)
    y = coords_col[:, 1:2].detach().requires_grad_(True)

    out = model(torch.cat([x, y], dim=1))
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

    x_scale, y_scale = scales

    u_x = x_scale * u_x_norm  # ∂/∂x = 2 / (x_max - x_min) · ∂/∂x_norm
    u_y = y_scale * u_y_norm  # ∂/∂y = 2 / (y_max - y_min) · ∂/∂y_norm

    v_x = x_scale * v_x_norm
    v_y = y_scale * v_y_norm

    p_x = x_scale * p_x_norm
    p_y = y_scale * p_y_norm

    u_xx = x_scale**2 * u_xx_norm
    u_yy = y_scale**2 * u_yy_norm

    v_xx = x_scale**2 * v_xx_norm
    v_yy = y_scale**2 * v_yy_norm

    res_mom_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    res_mom_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    res_cont = u_x + v_y

    return res_mom_x, res_mom_y, res_cont



def NS2Dtime(
        model: torch.nn.Module,
        coords_col: torch.Tensor,
        pde_terms: torch.Tensor,
        scales: tuple[float, float, float],
        nu: float,
    ) -> torch.Tensor:

    x = coords_col[:, 0:1].detach().requires_grad_(True)
    y = coords_col[:, 1:2].detach().requires_grad_(True)
    t = coords_col[:, 2:3].detach().requires_grad_(True)
    f_x = pde_terms[:, 0:1].detach().requires_grad_(False)
    f_y = pde_terms[:, 1:2].detach().requires_grad_(False)  
    nu_t = pde_terms[:, 2:3].detach().requires_grad_(False)  

    out = model(torch.cat([x, y, t], dim=1))
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    # calculate gradients, scaled by the spatial scales(from normalization)

    u_x_norm, u_y_norm, u_t_norm = torch.autograd.grad(
        u, [x, y, t], grad_outputs=torch.ones_like(u), create_graph=True
    )
    v_x_norm, v_y_norm, v_t_norm = torch.autograd.grad(
        v, [x, y, t], grad_outputs=torch.ones_like(u), create_graph=True
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

    x_scale, y_scale, t_scale = scales
    u_x = x_scale * u_x_norm  # ∂/∂x = 2 / (x_max - x_min) · ∂/∂x_norm
    u_y = y_scale * u_y_norm  # ∂/∂y = 2 / (y_max - y_min) · ∂/∂y_norm
    u_t = t_scale * u_t_norm  # ∂/∂t = 2 / (t_max - t_min) · ∂/∂t_norm

    v_x = x_scale * v_x_norm
    v_y = y_scale * v_y_norm
    v_t = t_scale * v_t_norm

    p_x = x_scale * p_x_norm
    p_y = y_scale * p_y_norm

    u_xx = x_scale**2 * u_xx_norm
    u_yy = y_scale**2 * u_yy_norm

    v_xx = x_scale**2 * v_xx_norm
    v_yy = y_scale**2 * v_yy_norm

    res_mom_x = u_t + u * u_x + v * u_y + p_x - (nu + nu_t) * (u_xx + u_yy) - f_x #rho bei kraft und pressure wahrscheinlich schon eingerechten, prüfen!
    res_mom_y = v_t + u * v_x + v * v_y + p_y - (nu + nu_t) * (v_xx + v_yy) - f_y #hier ist nu_t keine konstante mehr, darf also eigentlich nicht rausgezogen werden
    res_cont = u_x + v_y

    return res_mom_x, res_mom_y, res_cont
