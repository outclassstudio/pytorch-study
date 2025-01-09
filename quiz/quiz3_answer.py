import torch

def test_problem_1(user_input):
    t1 = torch.tensor(user_input)
    assert t1[0, 1].item() == 2, "문제 1-1 실패"
    assert t1[2, 0].item() == 7, "문제 1-2 실패"
    print("문제 1 통과")

def test_problem_2(user_input):
    t2 = torch.tensor(user_input)
    assert torch.equal(t2[0:2, :], torch.tensor([[1, 2, 3], [4, 5, 6]])), "문제 2-1 실패"
    assert torch.equal(t2[:, 0], torch.tensor([1, 4, 7])), "문제 2-2 실패"
    print("문제 2 통과")

def test_problem_3(user_input):
    t3 = torch.tensor(user_input)
    assert torch.equal(t3[-3:], torch.tensor([4, 5, 6])), "문제 3 실패"
    print("문제 3 통과")

def test_problem_4(user_input):
    t4 = torch.tensor(user_input)
    assert torch.equal(t4[t4 > 0], torch.tensor([1, 3, 5, 7, 9])), "문제 4-1 실패"
    assert torch.equal(t4[t4 % 3 == 0], torch.tensor([3, 9])), "문제 4-2 실패"
    print("문제 4 통과")

def test_problem_5(user_input):
    t5 = torch.tensor(user_input)
    t5[1, :] = 0
    assert torch.equal(t5, torch.tensor([[1, 2, 3], [0, 0, 0], [7, 8, 9]])), "문제 5 실패"
    print("문제 5 통과")

def test_problem_6(user_input):
    t6 = torch.tensor(user_input)
    assert torch.equal(t6[[0, 1, 2], [1, 2, 0]], torch.tensor([2, 6, 7])), "문제 6-1 실패"
    assert torch.equal(t6[1:3, [0, 2]], torch.tensor([[4, 6], [7, 9]])), "문제 6-2 실패"
    print("문제 6 통과")

def test_problem_7(user_input):
    t7 = torch.tensor(user_input)
    t7[1, 1, 2] = 42
    t7[:, -1, :2] = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    assert t7[1, 1, 2].item() == 42, "문제 7-1 실패"
    assert torch.equal(t7[:, -1, :2], torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])), "문제 7-2 실패"
    print("문제 7 통과")

def test_problem_8(user_input):
    t8 = torch.tensor(user_input)
    t8[..., :2] = 0
    assert torch.equal(t8[..., :2], torch.zeros(3, 4, 2)), "문제 8 실패"
    print("문제 8 통과")

def test_problem_9(user_input):
    t9 = torch.tensor(user_input)
    assert torch.equal(t9[1:3, 1:3], torch.tensor([[6, 7], [10, 11]])), "문제 9 실패"
    print("문제 9 통과")

def test_problem_10(user_input):
    t10 = torch.tensor(user_input)
    assert torch.equal(t10[:2][t10[:2] > 0], torch.tensor([1, 3, 4, 6])), "문제 10-1 실패"
    assert torch.equal(t10[(t10 > 0) & (t10 % 2 == 0)], torch.tensor([4, 6, 8])), "문제 10-2 실패"
    print("문제 10 통과")


if __name__ == "__main__":
    test_problem_1()
    test_problem_2()
    test_problem_3()
    test_problem_4()
    test_problem_5()
    test_problem_6()
    test_problem_7()
    test_problem_8()
    test_problem_9()
    test_problem_10()
