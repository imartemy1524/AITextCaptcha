import time
import cv2
import numpy as np
import onnxruntime
import requests
from fuzzywuzzy import fuzz
from matplotlib import pyplot as plt, image as mpimg
from utils.config import characters, img_height, img_width, img_type, data_dir_test, max_length
from utils.tester import mistakes_analyzer

real_tests = False

data_dir_test = data_dir_test.glob(img_type)
sess = onnxruntime.InferenceSession(r"../out.model.onnx")
name = sess.get_inputs()[0].name

def get_result(pred):
    """CTC decoder of the output tensor
    https://distill.pub/2017/ctc/
    https://en.wikipedia.org/wiki/Connectionist_temporal_classification
    :return string, float
    """
    accuracy = 1
    last = None
    ans = []
    # pred - 3d tensor, we need 2d array - first element
    for item in pred[0]:
        # get index of element with max accuracy
        char_ind = item.argmax()
        # ignore duplicates and special characters
        if char_ind != last and char_ind != 0 and char_ind != len(characters)+1:
            # this element is a character - append it to answer
            ans.append(characters[char_ind - 1])
            # Get accuracy for current character and multiply global accuracy by it
            accuracy *= item[char_ind]
        last = char_ind

    answ = "".join(ans)[:max_length]
    return answ, accuracy


def decode_img(data_bytes: bytes):
    # same actions, as for tensorflow
    image = cv2.imdecode(np.asarray(bytearray(data_bytes), dtype=np.uint8), -1)
    image: "np.ndarray" = image.astype(np.float32) / 255.
    if image.shape != (img_height, img_width, 3):
        cv2.resize(image, (img_width, img_height))
    image = image.transpose([1, 0, 2])
    #  Creating tensor ( adding 4d dimension )
    image = np.array([image])
    return image


def solve(file_name: str):
    with open(file_name, 'rb') as F:
        data_bytes = F.read()
    img = decode_img(data_bytes)

    pred_onx = sess.run(None, {name: img})[0]
    ans = get_result(pred_onx)
    return ans


if real_tests:
    time_now = time.time()
    _, ax = plt.subplots(4, 4, figsize=(10, 5))
    for i in range(16):
        file = 'image.jpeg'
        perc_ = 0
        while perc_ < 0.3:
            image = requests.get('https://api.vk.com/captcha.php?sid=98483')
            with open(file, 'wb') as f:
                f.write(image.content)
            solution, perc_ = solve(file)
        img = mpimg.imread(file)

        title = f"{perc_: .2%} {solution}"
        ax[i >> 2, i % 4].imshow(img)
        ax[i >> 2, i % 4].set_title(title)
        ax[i >> 2, i % 4].axis("off")
    plt.show()
    print("ARGV:", (time.time() - time_now) / 10, "sec.")

else:
    mistakes = []
    total = 0
    correct = 0
    for file in data_dir_test:
        ans = file.name.split(".")[0].split('-')[0]
        soved_ans, _ = solve(str(file))
        if soved_ans == ans:
            correct += 1
        else:
            mistakes.append((ans, soved_ans))
        total += 1
        if total % 100 == 0:
            print(f"Success: {correct / total:.2%}", end='\r')
    print(f"\n\nSuccess: {correct / total:.2%}", end='\n\n')
    print("Parsing data mistakes: ")
    anylize = mistakes_analyzer(mistakes)
    print("\n\nLoose symbols:")
    for i, v in sorted(anylize[0].items(), key=lambda n: -n[1]):
        if v == 1: continue  # 1 is not an error - mistake
        print(i, '=', v)
    print("\n\nWrong symbols:")
    for symbol, arr in sorted(anylize[1].items(), key=lambda n: -sum(n[1].values())):
        print(symbol, '=', sum(arr.values()), ":")
        for symbol_next, v in sorted(arr.items(), key=lambda n: -n[1]):
            if v == 1: continue
            print(f'\t{symbol} - {symbol_next} = {v}')
    for i in anylize[2]:
        if fuzz.ratio(*i) < 0.4:
            print("Probably problem in data: ")
            print(*i)
