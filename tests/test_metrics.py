from metrics import score, sentence_cer


def test_identical():
    S = "njan nale varum ente veedu evide aanu"
    r = score([S], [S])
    assert r["CER"] == 0 and r["WER"] == 0 and r["BLEU"] == 100 and r["chrF"] == 100


def test_completely_different():
    r = score(["zzzz qqqq"], ["njan nale varum ente veedu evide"])
    assert r["CER"] > 80 and r["WER"] == 100 and r["BLEU"] < 5 and r["chrF"] < 10


def test_known_cer_wer():
    # one wrong char out of 7 (a-b-c-d-e-f-g vs a-b-c-d-e-f-x), one wrong word out of 1
    r = score(["abcdefx"], ["abcdefg"])
    assert abs(r["CER"] - 100 / 7) < 0.01 and r["WER"] == 100


def test_empty_prediction_is_full_error():
    assert score([""], ["abc"])["CER"] == 100
    assert sentence_cer("", "abc") == 100


def test_directions():
    S = "njan nale varum ente veedu evide aanu"
    good = score([S], [S])
    bad = score(["njaan nal varu ente veed evid aanu"], [S])
    assert bad["CER"] > good["CER"] and bad["WER"] > good["WER"] and bad["BLEU"] < good["BLEU"] and bad["chrF"] < good["chrF"]
