from enum import Enum
import numpy as np

class Rules(Enum):
    """
    Enum để xác định các quy tắc điều độ. Các giá trị tương ứng với không gian hành động.
    """
    SPT = 0   # Thời gian xử lý ngắn nhất (Shortest Processing Time)
    LPT = 1   # Thời gian xử lý dài nhất (Longest Processing Time)
    STPT = 2  # Tổng thời gian xử lý ngắn nhất (cho công việc) (Shortest Total Processing Time)
    LTPT = 3  # Tổng thời gian xử lý dài nhất (cho công việc) (Longest Total Processing Time)
    LOR = 4   # Ít hoạt động còn lại nhất (cho công việc) (Least Operations Remaining)
    MOR = 5   # Nhiều hoạt động còn lại nhất (cho công việc) (Most Operations Remaining)
    LQNO = 6  # Hàng đợi ngắn nhất tại máy của hoạt động tiếp theo (Least Queue at Next Operation's machine)
    MQNO = 7  # Hàng đợi dài nhất tại máy của hoạt động tiếp theo (Most Queue at Next Operation's machine)

def get_job_and_op_indices(op_id, env):
    """Hàm hỗ trợ để lấy chỉ số công việc (hàng) và hoạt động (cột) từ một ID hoạt động."""
    return op_id // env.number_of_machines, op_id % env.number_of_machines

def apply_dispatching_rule(rule, candidates, env):
    """
    Chọn hoạt động tốt nhất từ danh sách các ứng viên dựa trên một quy tắc.
    
    Args:
        rule (Rules): Quy tắc điều độ để áp dụng.
        candidates (np.ndarray): Một mảng numpy chứa các ID hoạt động hợp lệ.
        env (SJSSP): Đối tượng môi trường, để truy cập thông tin trạng thái.

    Returns:
        int: ID của hoạt động được chọn.
    """
    if len(candidates) == 0:
        raise ValueError("Không thể áp dụng quy tắc cho một tập hợp ứng viên rỗng.")
    
    if len(candidates) == 1:
        return candidates[0]

    if rule == Rules.SPT:
        candidate_durations = env.dur.flatten()[candidates]
        best_candidate_idx = np.argmin(candidate_durations)
        return candidates[best_candidate_idx]

    if rule == Rules.LPT:
        candidate_durations = env.dur.flatten()[candidates]
        best_candidate_idx = np.argmax(candidate_durations)
        return candidates[best_candidate_idx]

    if rule == Rules.STPT:
        job_total_times = []
        for op_id in candidates:
            job_idx, _ = get_job_and_op_indices(op_id, env)
            total_time = np.sum(env.dur[job_idx, :])
            job_total_times.append(total_time)
        best_candidate_idx = np.argmin(job_total_times)
        return candidates[best_candidate_idx]

    if rule == Rules.LTPT:
        job_total_times = []
        for op_id in candidates:
            job_idx, _ = get_job_and_op_indices(op_id, env)
            total_time = np.sum(env.dur[job_idx, :])
            job_total_times.append(total_time)
        best_candidate_idx = np.argmax(job_total_times)
        return candidates[best_candidate_idx]

    if rule == Rules.LOR:
        ops_remaining = []
        for op_id in candidates:
            _, op_idx = get_job_and_op_indices(op_id, env)
            remaining = env.number_of_machines - op_idx
            ops_remaining.append(remaining)
        best_candidate_idx = np.argmin(ops_remaining)
        return candidates[best_candidate_idx]

    if rule == Rules.MOR:
        ops_remaining = []
        for op_id in candidates:
            _, op_idx = get_job_and_op_indices(op_id, env)
            remaining = env.number_of_machines - op_idx
            ops_remaining.append(remaining)
        best_candidate_idx = np.argmax(ops_remaining)
        return candidates[best_candidate_idx]



    if rule == Rules.LQNO or rule == Rules.MQNO:
        machine_queues = {}
        all_eligible_ops = env.omega[~env.mask] 
        for op_id in all_eligible_ops:
            job_idx, op_idx = get_job_and_op_indices(op_id, env)
            machine_needed = env.m[job_idx, op_idx]
            machine_queues[machine_needed] = machine_queues.get(machine_needed, 0) + 1

        look_ahead_scores = [] 
        for op_id in candidates:
            job_idx, op_idx = get_job_and_op_indices(op_id, env)
            
            if op_idx == env.number_of_machines - 1:
                look_ahead_scores.append(0)
            else:
                next_op_machine = env.m[job_idx, op_idx + 1]
                queue_size = machine_queues.get(next_op_machine, 0)
                look_ahead_scores.append(queue_size)
        
        if rule == Rules.LQNO:
            best_candidate_idx = np.argmin(look_ahead_scores)
            return candidates[best_candidate_idx]
        else: 
            best_candidate_idx = np.argmax(look_ahead_scores)
            return candidates[best_candidate_idx]

    raise NotImplementedError(f"Quy tắc điều độ {rule} chưa được triển khai.")