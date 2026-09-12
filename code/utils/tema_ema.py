import torch


def detach_model(model):
    for p in model.parameters():
        p.detach_()
        p.requires_grad_(False)


@torch.no_grad()
def copy_student_to_teacher(student, teacher):
    teacher.load_state_dict(student.state_dict())


@torch.no_grad()
def update_ema_variables(student, teacher, alpha, global_step=None):
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1.0 - alpha)
    for ema_buf, buf in zip(teacher.buffers(), student.buffers()):
        ema_buf.copy_(buf)
