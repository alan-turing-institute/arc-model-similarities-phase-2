from modsim2.similarity.metrics import kde, mmd, otdd, pad

ARGUMENTS = "arguments"
CLASS_KEY = "class"
METRIC_CLS_DICT = {
    "mmd": mmd.MMD,
    "otdd": otdd.OTDD,
    "pad": pad.PAD,
    "kde": kde.KDE,
}
