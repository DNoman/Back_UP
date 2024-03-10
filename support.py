def changing_learning_rate(initial_learning_rate, decay_rate, scores):
    # Tính toán trung bình của các scores
    avg_score = sum(scores) / len(scores)

    # Tính toán learning rate mới dựa trên decay rate và trung bình của các scores
    new_learning_rate = max(0.0001, initial_learning_rate - avg_score * decay_rate / 5)

    return new_learning_rate


# Sử dụng hàm để cập nhật learning rate
initial_learning_rate = 0.01  # Tốc độ học ban đầu
decay_rate = 0.001  # Hệ số giảm của learning rate
scores = [0.5, 0.6, 0.7, 0.8, 0.9]  # Điểm số của 5 phiên trước đó

new_learning_rate = changing_learning_rate(initial_learning_rate, decay_rate, scores)
print("New learning rate:", new_learning_rate)