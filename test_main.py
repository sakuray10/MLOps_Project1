from main import detectron
import os
import os.path

input_img1 = "cooking.jpeg"
input_img2 = "cookout.jpg"
input_img3 = "family.jpg"


def test_detectron():
    test = detectron(input_img1, "out1.jpg")
    assert os.path.isfile("out1.jpg")

    test2 = detectron(input_img2, "out2.jpg")
    assert os.path.isfile("out2.jpg")

    test3 = detectron(input_img3, "out3.jpg")
    assert os.path.isfile("out3.jpg")