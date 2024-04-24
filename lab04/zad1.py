import math


def forward_pass(wiek, waga, wzrost):
    hidden1 = wiek * -0.46122 + waga * 0.97314 + wzrost * -0.39203 + 0.80109
    hidden1_po_aktywacji = 1 / (1 + math.exp(-hidden1))
    hidden2 = wiek * 0.78548 + waga * 2.10584 + wzrost * -0.57847 + 0.43529
    hidden2_po_aktywacji = 1 / (1 + math.exp(-hidden2))
    output = hidden1_po_aktywacji * -0.81546 + hidden2_po_aktywacji * 1.03775 + -0.2368
    return output


print(forward_pass(25, 67, 180))
print(round(forward_pass(25, 67, 180)))

print(forward_pass(28, 120, 175))
print(round(forward_pass(28, 120, 175)))
