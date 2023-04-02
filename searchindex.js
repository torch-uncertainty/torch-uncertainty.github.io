Search.setIndex({"docnames": ["api", "auto_tutorials/index", "auto_tutorials/pe_cifar10_tutorial", "auto_tutorials/sg_execution_times", "generated/torch_uncertainty.baselines.PackedResNet", "generated/torch_uncertainty.layers.PackedConv2d", "generated/torch_uncertainty.layers.PackedLinear", "generated/torch_uncertainty.metrics.BrierScore", "generated/torch_uncertainty.metrics.Disagreement", "generated/torch_uncertainty.metrics.Entropy", "generated/torch_uncertainty.metrics.JensenShannonDivergence", "generated/torch_uncertainty.metrics.MutualInformation", "generated/torch_uncertainty.metrics.NegativeLogLikelihood", "generated/torch_uncertainty.models.resnet.masked_resnet101", "generated/torch_uncertainty.models.resnet.masked_resnet152", "generated/torch_uncertainty.models.resnet.masked_resnet18", "generated/torch_uncertainty.models.resnet.masked_resnet34", "generated/torch_uncertainty.models.resnet.masked_resnet50", "generated/torch_uncertainty.models.resnet.packed_resnet101", "generated/torch_uncertainty.models.resnet.packed_resnet152", "generated/torch_uncertainty.models.resnet.packed_resnet18", "generated/torch_uncertainty.models.resnet.packed_resnet34", "generated/torch_uncertainty.models.resnet.packed_resnet50", "generated/torch_uncertainty.models.resnet.resnet101", "generated/torch_uncertainty.models.resnet.resnet152", "generated/torch_uncertainty.models.resnet.resnet18", "generated/torch_uncertainty.models.resnet.resnet34", "generated/torch_uncertainty.models.resnet.resnet50", "index", "installation", "quickstart"], "filenames": ["api.rst", "auto_tutorials/index.rst", "auto_tutorials/pe_cifar10_tutorial.rst", "auto_tutorials/sg_execution_times.rst", "generated/torch_uncertainty.baselines.PackedResNet.rst", "generated/torch_uncertainty.layers.PackedConv2d.rst", "generated/torch_uncertainty.layers.PackedLinear.rst", "generated/torch_uncertainty.metrics.BrierScore.rst", "generated/torch_uncertainty.metrics.Disagreement.rst", "generated/torch_uncertainty.metrics.Entropy.rst", "generated/torch_uncertainty.metrics.JensenShannonDivergence.rst", "generated/torch_uncertainty.metrics.MutualInformation.rst", "generated/torch_uncertainty.metrics.NegativeLogLikelihood.rst", "generated/torch_uncertainty.models.resnet.masked_resnet101.rst", "generated/torch_uncertainty.models.resnet.masked_resnet152.rst", "generated/torch_uncertainty.models.resnet.masked_resnet18.rst", "generated/torch_uncertainty.models.resnet.masked_resnet34.rst", "generated/torch_uncertainty.models.resnet.masked_resnet50.rst", "generated/torch_uncertainty.models.resnet.packed_resnet101.rst", "generated/torch_uncertainty.models.resnet.packed_resnet152.rst", "generated/torch_uncertainty.models.resnet.packed_resnet18.rst", "generated/torch_uncertainty.models.resnet.packed_resnet34.rst", "generated/torch_uncertainty.models.resnet.packed_resnet50.rst", "generated/torch_uncertainty.models.resnet.resnet101.rst", "generated/torch_uncertainty.models.resnet.resnet152.rst", "generated/torch_uncertainty.models.resnet.resnet18.rst", "generated/torch_uncertainty.models.resnet.resnet34.rst", "generated/torch_uncertainty.models.resnet.resnet50.rst", "index.rst", "installation.rst", "quickstart.rst"], "titles": ["API reference", "Tutorials", "From a Vanilla Classifier to a Packed-Ensemble", "Computation times", "PackedResNet", "PackedConv2d", "PackedLinear", "BrierScore", "Disagreement", "Entropy", "JensenShannonDivergence", "MutualInformation", "NegativeLogLikelihood", "torch_uncertainty.models.resnet.masked_resnet101", "torch_uncertainty.models.resnet.masked_resnet152", "torch_uncertainty.models.resnet.masked_resnet18", "torch_uncertainty.models.resnet.masked_resnet34", "torch_uncertainty.models.resnet.masked_resnet50", "torch_uncertainty.models.resnet.packed_resnet101", "torch_uncertainty.models.resnet.packed_resnet152", "torch_uncertainty.models.resnet.packed_resnet18", "torch_uncertainty.models.resnet.packed_resnet34", "torch_uncertainty.models.resnet.packed_resnet50", "torch_uncertainty.models.resnet.resnet101", "torch_uncertainty.models.resnet.resnet152", "torch_uncertainty.models.resnet.resnet18", "torch_uncertainty.models.resnet.resnet34", "torch_uncertainty.models.resnet.resnet50", "Torch Uncertainty", "Installation", "Quickstart"], "terms": {"resnet": [0, 4, 30], "pack": [0, 1, 3, 4, 5, 6, 18, 19, 20, 21, 22, 28, 30], "ensembl": [0, 1, 3, 4, 5, 6, 8, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 28, 30], "masksembl": [0, 13, 14, 15, 16, 17], "below": [1, 30], "i": [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 30], "galleri": [1, 2], "exampl": [1, 2, 4, 30], "from": [1, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30], "vanilla": [1, 3], "classifi": [1, 3], "download": [1, 2], "all": [1, 2, 5], "python": [1, 2, 4, 29], "sourc": [1, 2, 28], "code": [1, 2, 30], "auto_tutorials_python": 1, "zip": 1, "jupyt": [1, 2], "notebook": [1, 2], "auto_tutorials_jupyt": 1, "gener": [1, 2], "sphinx": [1, 2], "go": 2, "end": 2, "full": [2, 30], "let": [2, 30], "": [2, 4, 7, 30], "dive": 2, "step": 2, "process": [2, 29], "modifi": 2, "In": [2, 30], "thi": [2, 5, 6], "tutori": [2, 28], "we": 2, "us": [2, 4, 5, 6, 28, 29], "avail": [2, 29, 30], "torchvis": 2, "packag": [2, 29], "The": [2, 4, 6, 7, 8, 9, 10, 11, 12, 29, 30], "consist": 2, "60000": 2, "32x32": 2, "colour": 2, "10": [2, 30], "class": [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "6000": 2, "per": [2, 5, 7, 8, 9, 10, 11, 12], "There": 2, "ar": [2, 5, 6, 7, 8, 10, 11, 12, 29, 30], "50000": 2, "10000": 2, "here": [2, 28, 29], "an": [2, 8, 10, 11, 30], "what": [2, 4], "look": 2, "like": [2, 30], "outlin": 2, "evalu": 2, "its": [2, 30], "perform": 2, "w": 2, "r": 2, "t": [2, 7], "uncertainti": [2, 10, 11, 29, 30], "quantif": [2, 28], "ood": [2, 4], "detect": 2, "import": [2, 30], "torch": [2, 4, 7, 8, 9, 10, 11, 12, 29, 30], "transform": 2, "output": [2, 5, 6], "pilimag": 2, "rang": 2, "0": [2, 3, 5, 7, 30], "them": 2, "tensor": [2, 7, 8, 9, 10, 11, 12], "If": [2, 4, 5, 7, 8, 9, 10, 11, 12, 29], "run": 2, "window": 2, "you": [2, 5, 6, 29, 30], "get": 2, "brokenpipeerror": 2, "try": 2, "set": [2, 4, 7, 8, 9, 10, 11, 12, 30], "num_work": 2, "util": 2, "dataload": 2, "compos": 2, "totensor": 2, "batch_siz": [2, 6], "trainset": 2, "root": [2, 30], "true": [2, 4, 5, 6], "trainload": 2, "shuffl": 2, "testset": 2, "fals": [2, 4], "testload": 2, "plane": 2, "car": 2, "bird": 2, "cat": 2, "deer": 2, "dog": 2, "frog": 2, "hors": 2, "ship": 2, "truck": 2, "http": [2, 29], "www": 2, "c": [2, 7, 8, 9, 10, 11, 12], "toronto": 2, "edu": 2, "kriz": 2, "cifar": [2, 30], "tar": 2, "gz": 2, "170498071": 2, "00": 2, "98304": 2, "04": [2, 3], "06": 2, "690722": 2, "05it": 2, "458752": 2, "01": 2, "19": 2, "2137395": 2, "26it": 2, "1900544": 2, "23": 2, "7236816": 2, "62it": 2, "6553600": 2, "07": 2, "21817874": 2, "47it": 2, "8": 2, "7": 2, "12812288": 2, "35097870": 2, "96it": 2, "11": 2, "19005440": 2, "03": 2, "43051982": 2, "76it": 2, "15": 2, "25165824": 2, "47866446": 2, "12it": 2, "18": [2, 4, 15, 20, 25], "31358976": 2, "02": 2, "52142390": 2, "04it": 2, "22": 2, "36962304": 2, "53250935": 2, "69it": 2, "25": 2, "43122688": 2, "54740085": 2, "54it": 2, "29": 2, "49315840": 2, "55873138": 2, "37it": 2, "33": 2, "55476224": 2, "57439636": 2, "42it": 2, "36": 2, "6": 2, "61702144": 2, "58746829": 2, "50it": 2, "40": 2, "9": 2, "67633152": 2, "58898070": 2, "22it": 2, "43": 2, "73531392": 2, "58584921": 2, "46it": 2, "47": 2, "79396864": 2, "58311653": 2, "28it": 2, "50": [2, 4, 17, 22, 27], "85327872": 2, "58595777": 2, "85it": 2, "53": 2, "91193344": 2, "57975859": 2, "60it": 2, "57": 2, "97026048": 2, "57678119": 2, "65it": 2, "60": 2, "103120896": 2, "58635910": 2, "48it": 2, "64": [2, 5], "109346816": 2, "59683982": 2, "68": 2, "115507200": 2, "60206068": 2, "52it": 2, "71": 2, "121536512": 2, "59514685": 2, "73it": 2, "75": 2, "127500288": 2, "58718508": 2, "44it": 2, "78": 2, "133398528": 2, "58240427": 2, "93it": 2, "82": 2, "139231232": 2, "58235045": 2, "94it": 2, "85": 2, "145063936": 2, "57764159": 2, "10it": 2, "88": 2, "150863872": 2, "57605170": 2, "92": 2, "156958720": 2, "58408864": 2, "90it": 2, "96": 2, "163184640": 2, "59381344": 2, "64it": 2, "99": 2, "169181184": 2, "59543766": 2, "100": [2, 30], "52640602": 2, "40it": 2, "extract": 2, "file": [2, 3, 30], "alreadi": [2, 30], "verifi": 2, "u": 2, "show": 2, "some": 2, "fun": 2, "matplotlib": 2, "pyplot": 2, "plt": 2, "numpi": 2, "np": 2, "def": 2, "imshow": 2, "img": 2, "unnorm": 2, "npimg": 2, "transpos": 2, "random": 2, "datait": 2, "iter": 2, "label": [2, 12], "next": 2, "make_grid": 2, "print": 2, "join": 2, "f": 2, "j": 2, "first": 2, "refer": [2, 28], "convolut": [2, 5, 6, 23, 24, 25, 26, 27], "neural": 2, "network": 2, "nn": [2, 4, 30], "net": 2, "modul": [2, 4, 28, 30], "__init__": 2, "self": 2, "super": 2, "conv1": 2, "conv2d": [2, 5], "pool": 2, "maxpool2d": 2, "conv2": 2, "16": 2, "fc1": 2, "linear": [2, 6], "120": 2, "fc2": 2, "84": 2, "fc3": 2, "forward": [2, 30], "x": 2, "relu": 2, "flatten": 2, "return": [2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "paramet": [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "m": 2, "alpha": [2, 4, 18, 19, 20, 21, 22], "text": [2, 5, 6], "gamma": [2, 4, 18, 19, 20, 21, 22], "einop": 2, "rearrang": [2, 6], "torch_uncertainti": [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 30], "layer": [2, 5, 6, 28, 30], "packedconv2d": 2, "packedlinear": 2, "packednet": 2, "none": [2, 5, 6, 7, 8, 9, 10, 11, 12], "left": 2, "sinc": 2, "subnetwork": 2, "have": 2, "input": [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "num_estim": [2, 4, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], "group": [2, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "e": 2, "h": 2, "packed_net": 2, "classif": 2, "cross": 2, "entropi": [2, 4], "sgd": 2, "momentum": 2, "criterion": [2, 4], "crossentropyloss": [2, 30], "lr": 2, "001": 2, "epoch": 2, "loop": 2, "over": [2, 7, 8, 9, 10, 11, 12], "multipl": 2, "time": [2, 5, 6], "running_loss": 2, "enumer": 2, "list": 2, "zero": 2, "gradient": 2, "zero_grad": 2, "backward": 2, "repeat": 2, "statist": 2, "item": 2, "2000": 2, "1999": 2, "everi": [2, 29], "mini": 2, "batch": [2, 7, 8, 9, 10, 11, 12], "5d": 2, "3f": 2, "finish": 2, "303": 2, "4000": 2, "278": 2, "251": 2, "8000": 2, "184": 2, "075": 2, "12000": 2, "007": 2, "896": 2, "822": 2, "782": 2, "717": 2, "692": 2, "674": 2, "save": 2, "our": 2, "model": [2, 4, 8, 9, 10, 12, 28, 30], "path": [2, 30], "cifar_packed_net": 2, "pth": 2, "state_dict": 2, "displai": 2, "familiar": 2, "groundtruth": 2, "back": 2, "note": [2, 6], "re": 2, "wasn": 2, "necessari": [2, 6], "onli": [2, 4, 5, 6, 30], "did": 2, "illustr": 2, "how": [2, 7, 8, 9, 10, 11, 12], "do": [2, 30], "so": [2, 5, 6, 29, 30], "load_state_dict": 2, "kei": 2, "match": 2, "successfulli": 2, "see": [2, 5, 6, 7, 8, 9, 10, 11, 12], "think": 2, "abov": 2, "logit": [2, 4], "n": [2, 7, 8, 9, 10, 11, 29], "b": [2, 7, 8, 9, 10, 11, 12], "probs_per_est": 2, "softmax": 2, "dim": 2, "mean": [2, 7, 8, 9, 10, 11, 12], "_": 2, "predict": [2, 4, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "max": 2, "result": 2, "seem": 2, "pretti": 2, "good": 2, "total": [2, 3], "script": [2, 4], "minut": 2, "651": [2, 3], "second": 2, "pe_cifar10_tutori": [2, 3], "py": [2, 3, 4, 30], "ipynb": 2, "05": 3, "execut": [3, 6, 29, 30], "auto_tutori": 3, "mb": 3, "baselin": [4, 28, 30], "num_class": [4, 7, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "in_channel": [4, 5, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "arch": 4, "loss": [4, 30], "optimization_procedur": [4, 30], "use_entropi": 4, "use_logit": 4, "use_mi": 4, "use_variation_ratio": 4, "kwarg": [4, 7, 8, 9, 10, 11, 12], "lightningmodul": 4, "int": [4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "number": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "estim": [4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 28], "channel": [4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "expans": [4, 18, 19, 20, 21, 22], "factor": [4, 18, 19, 20, 21, 22], "affect": [4, 18, 19, 20, 21, 22], "width": [4, 18, 19, 20, 21, 22], "within": [4, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], "each": [4, 5, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22], "determin": [4, 7, 8, 9, 10, 11, 12], "which": [4, 30], "architectur": [4, 30], "32": 4, "101": [4, 13, 18, 23], "152": [4, 14, 19, 24], "train": [4, 30], "ani": 4, "optim": [4, 30], "procedur": 4, "correspond": [4, 30], "expect": 4, "configure_optim": 4, "method": 4, "bool": [4, 5, 6], "option": [4, 5, 6, 7, 8, 9, 10, 11, 12], "indic": 4, "whether": 4, "valu": [4, 7], "default": [4, 5, 6, 7, 8, 9, 10, 11, 12], "mutual": [4, 11], "inform": [4, 11], "variat": 4, "ratio": 4, "confid": [4, 8, 9], "score": [4, 7, 8, 9, 10, 11, 12], "make": [4, 5, 6, 7, 8, 10, 11, 12], "sure": [4, 5, 6, 7, 8, 10, 11, 12], "most": 4, "one": [4, 7, 8, 9, 10, 11, 12], "attribut": 4, "otherwis": 4, "valueerror": [4, 7, 8, 9, 10, 11, 12], "rais": [4, 7, 8, 9, 10, 11, 12], "1": [4, 5, 6, 7, 30], "static": 4, "add_model_specific_arg": 4, "parent_pars": 4, "defin": 4, "via": [4, 29, 30], "command": [4, 29], "line": 4, "mutual_inform": 4, "variation_ratio": 4, "4": 4, "2": 4, "out_channel": 5, "kernel_s": 5, "stride": 5, "pad": 5, "dilat": 5, "minimum_channels_per_group": 5, "bia": [5, 6], "devic": [5, 6, 30], "dtype": [5, 6], "style": [5, 6, 13, 14, 15, 16, 17, 30], "imag": [5, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "produc": [5, 6], "tupl": 5, "size": [5, 6, 7, 8, 9, 10, 11, 12], "convolv": 5, "kernel": 5, "str": [5, 7, 8, 9, 10, 11, 12], "ad": 5, "four": 5, "side": 5, "space": 5, "between": 5, "element": 5, "block": [5, 6], "connexion": 5, "smallest": 5, "possibl": 5, "hannel": 5, "add": [5, 6, 29, 30], "learnabl": [5, 6], "member": [5, 6], "frac": [5, 6, 7], "when": [5, 6], "should": [5, 6, 30], "both": [5, 6], "divis": [5, 6], "howev": [5, 6], "chang": [5, 6], "compli": [5, 6], "constraint": [5, 6], "in_featur": 6, "out_featur": 6, "comput": [6, 7, 8, 9, 10, 11, 12], "fulli": 6, "connect": 6, "oper": 6, "given": 6, "1x1": 6, "featur": 6, "It": 6, "compat": 6, "previou": 6, "later": 6, "n_estim": 6, "often": 6, "metric": [7, 8, 9, 10, 11, 12, 28, 30], "reduct": [7, 8, 9, 10, 11, 12], "brier": 7, "reduc": [7, 8, 9, 10, 11, 12], "dimens": [7, 8, 9, 10, 11, 12], "averag": [7, 8, 9, 10, 11, 12], "across": [7, 8, 9, 10, 11, 12], "sampl": [7, 8, 9, 10, 11, 12], "sum": [7, 8, 9, 10, 11, 12], "addit": [7, 8, 9, 10, 11, 12], "keyword": [7, 8, 9, 10, 11, 12], "argument": [7, 8, 9, 10, 11, 12, 30], "advanc": [7, 8, 9, 10, 11, 12], "prob": [7, 8, 9, 10, 11, 12], "target": [7, 8, 12], "where": [7, 8, 9, 10, 11, 12], "3d": 7, "ie": 7, "sum_": 7, "probabl": [7, 8, 9, 10, 11, 12], "normal": [7, 8, 10, 11, 12], "final": [7, 30], "base": [7, 8, 9, 10, 11, 12], "pass": [7, 8, 9, 10, 11, 12], "updat": [7, 8, 9, 10, 11, 12], "type": [7, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "current": [7, 9, 11], "new": [7, 9, 10, 11, 29], "A": [7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28], "shape": 7, "higher": [8, 9, 10, 11], "lower": [8, 9], "state": [8, 10, 12], "shannon": [9, 10], "singl": 9, "accross": 9, "previous": [9, 11, 12], "jensen": 10, "diverg": 10, "epistem": [10, 11], "data": [10, 30], "neg": 12, "log": [12, 30], "likelihood": 12, "nll": 12, "ground": 12, "truth": 12, "scale": [13, 14, 15, 16, 17], "deep": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "residu": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "learn": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "recognit": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "_maskedresnet": [13, 14, 15, 16, 17], "34": [16, 21, 26], "_packedresnet": [18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "pytorch": [28, 30], "librari": [28, 30], "benchmark": 28, "leverag": 28, "effici": 28, "techniqu": 28, "offici": 28, "implement": 28, "paper": 28, "author": 28, "olivi": 28, "laurent": 28, "adrien": 28, "lafag": 28, "enzo": 28, "tartaglion": 28, "geoffrei": 28, "daniel": 28, "jean": 28, "marc": 28, "martinez": 28, "andrei": 28, "bursuc": 28, "gianni": 28, "franchi": 28, "instal": 28, "pypi": 28, "quickstart": 28, "cli": 28, "tool": 28, "your": [28, 29], "own": 28, "trainer": 28, "api": 28, "index": 28, "search": 28, "page": 28, "can": [29, 30], "pip": 29, "poetri": 29, "guidelin": 29, "thei": 29, "boil": 29, "down": 29, "follow": [29, 30], "curl": 29, "ssl": 29, "org": 29, "python3": [29, 30], "clone": 29, "repositori": 29, "git": 29, "github": 29, "com": 29, "ensta": 29, "u2i": 29, "creat": [29, 30], "conda": 29, "environ": 29, "activ": 29, "develop": 29, "dev": 29, "depend": 29, "system": 29, "mai": [29, 30], "encount": 29, "error": 29, "kill": 29, "python_keyring_backend": 29, "keyr": 29, "backend": 29, "null": 29, "begin": 29, "differ": 30, "level": 30, "start": 30, "highest": 30, "usag": 30, "provid": 30, "fledg": 30, "directli": 30, "To": 30, "experi": 30, "folder": 30, "cli_main": 30, "routin": 30, "take": 30, "lightn": 30, "valid": 30, "test": 30, "logic": 30, "For": 30, "instanc": 30, "might": 30, "datamodul": 30, "again": 30, "imagenet": 30, "200": 30, "dictionari": 30, "contain": 30, "name": 30, "schedul": 30, "mani": 30, "move": 30, "directori": 30, "acceler": 30, "gpu": 30, "multi": 30, "etc": 30, "cifar10": 30, "pathlib": 30, "packedresnet": 30, "cifar10datamodul": 30, "optim_cifar10_resnet18": 30, "__file__": 30, "parent": 30, "absolut": 30, "replac": 30, "now": 30, "meantim": 30, "feel": 30, "free": 30, "reus": 30, "dataset": 30}, "objects": {"torch_uncertainty.baselines": [[4, 0, 1, "", "PackedResNet"]], "torch_uncertainty.baselines.PackedResNet": [[4, 1, 1, "", "add_model_specific_args"]], "torch_uncertainty.layers": [[5, 0, 1, "", "PackedConv2d"], [6, 0, 1, "", "PackedLinear"]], "torch_uncertainty.metrics": [[7, 0, 1, "", "BrierScore"], [8, 0, 1, "", "Disagreement"], [9, 0, 1, "", "Entropy"], [10, 0, 1, "", "JensenShannonDivergence"], [11, 0, 1, "", "MutualInformation"], [12, 0, 1, "", "NegativeLogLikelihood"]], "torch_uncertainty.metrics.BrierScore": [[7, 1, 1, "", "compute"], [7, 1, 1, "", "update"]], "torch_uncertainty.metrics.Disagreement": [[8, 1, 1, "", "compute"], [8, 1, 1, "", "update"]], "torch_uncertainty.metrics.Entropy": [[9, 1, 1, "", "compute"], [9, 1, 1, "", "update"]], "torch_uncertainty.metrics.JensenShannonDivergence": [[10, 1, 1, "", "compute"], [10, 1, 1, "", "update"]], "torch_uncertainty.metrics.MutualInformation": [[11, 1, 1, "", "compute"], [11, 1, 1, "", "update"]], "torch_uncertainty.metrics.NegativeLogLikelihood": [[12, 1, 1, "", "compute"], [12, 1, 1, "", "update"]], "torch_uncertainty.models.resnet": [[13, 2, 1, "", "masked_resnet101"], [14, 2, 1, "", "masked_resnet152"], [15, 2, 1, "", "masked_resnet18"], [16, 2, 1, "", "masked_resnet34"], [17, 2, 1, "", "masked_resnet50"], [18, 2, 1, "", "packed_resnet101"], [19, 2, 1, "", "packed_resnet152"], [20, 2, 1, "", "packed_resnet18"], [21, 2, 1, "", "packed_resnet34"], [22, 2, 1, "", "packed_resnet50"], [23, 2, 1, "", "resnet101"], [24, 2, 1, "", "resnet152"], [25, 2, 1, "", "resnet18"], [26, 2, 1, "", "resnet34"], [27, 2, 1, "", "resnet50"]]}, "objtypes": {"0": "py:class", "1": "py:method", "2": "py:function"}, "objnames": {"0": ["py", "class", "Python class"], "1": ["py", "method", "Python method"], "2": ["py", "function", "Python function"]}, "titleterms": {"api": 0, "refer": 0, "baselin": 0, "model": [0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "layer": 0, "metric": 0, "tutori": 1, "from": [2, 29], "vanilla": 2, "classifi": 2, "pack": 2, "ensembl": 2, "dataset": 2, "train": 2, "imag": 2, "1": 2, "load": 2, "normal": 2, "cifar10": 2, "2": 2, "defin": 2, "3": 2, "loss": 2, "function": 2, "optim": 2, "4": 2, "data": 2, "5": 2, "test": 2, "comput": 3, "time": 3, "packedresnet": 4, "packedconv2d": 5, "packedlinear": 6, "brierscor": 7, "disagr": 8, "entropi": 9, "jensenshannondiverg": 10, "mutualinform": 11, "negativeloglikelihood": 12, "torch_uncertainti": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "resnet": [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], "masked_resnet101": 13, "masked_resnet152": 14, "masked_resnet18": 15, "masked_resnet34": 16, "masked_resnet50": 17, "packed_resnet101": 18, "packed_resnet152": 19, "packed_resnet18": 20, "packed_resnet34": 21, "packed_resnet50": 22, "resnet101": 23, "resnet152": 24, "resnet18": 25, "resnet34": 26, "resnet50": 27, "torch": 28, "uncertainti": 28, "content": 28, "indic": 28, "tabl": 28, "instal": 29, "pypi": 29, "sourc": 29, "quickstart": 30, "us": 30, "cli": 30, "tool": 30, "procedur": 30, "exempl": 30, "your": 30, "own": 30, "trainer": 30}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 57}, "alltitles": {"API reference": [[0, "api-reference"]], "Baselines": [[0, "baselines"]], "Models": [[0, "models"]], "Layers": [[0, "layers"]], "Metrics": [[0, "metrics"]], "Tutorials": [[1, "tutorials"]], "From a Vanilla Classifier to a Packed-Ensemble": [[2, "from-a-vanilla-classifier-to-a-packed-ensemble"]], "Dataset": [[2, "dataset"]], "Training a image Packed-Ensemble classifier": [[2, "training-a-image-packed-ensemble-classifier"]], "1. Load and normalize CIFAR10": [[2, "load-and-normalize-cifar10"]], "2. Define a Packed-Ensemble from a vanilla classifier": [[2, "define-a-packed-ensemble-from-a-vanilla-classifier"]], "3. Define a Loss function and optimizer": [[2, "define-a-loss-function-and-optimizer"]], "4. Train the Packed-Ensemble on the training data": [[2, "train-the-packed-ensemble-on-the-training-data"]], "5. Test the Packed-Ensemble on the test data": [[2, "test-the-packed-ensemble-on-the-test-data"]], "Computation times": [[3, "computation-times"]], "PackedResNet": [[4, "packedresnet"]], "PackedConv2d": [[5, "packedconv2d"]], "PackedLinear": [[6, "packedlinear"]], "BrierScore": [[7, "brierscore"]], "Disagreement": [[8, "disagreement"]], "Entropy": [[9, "entropy"]], "JensenShannonDivergence": [[10, "jensenshannondivergence"]], "MutualInformation": [[11, "mutualinformation"]], "NegativeLogLikelihood": [[12, "negativeloglikelihood"]], "torch_uncertainty.models.resnet.masked_resnet101": [[13, "torch-uncertainty-models-resnet-masked-resnet101"]], "torch_uncertainty.models.resnet.masked_resnet152": [[14, "torch-uncertainty-models-resnet-masked-resnet152"]], "torch_uncertainty.models.resnet.masked_resnet18": [[15, "torch-uncertainty-models-resnet-masked-resnet18"]], "torch_uncertainty.models.resnet.masked_resnet34": [[16, "torch-uncertainty-models-resnet-masked-resnet34"]], "torch_uncertainty.models.resnet.masked_resnet50": [[17, "torch-uncertainty-models-resnet-masked-resnet50"]], "torch_uncertainty.models.resnet.packed_resnet101": [[18, "torch-uncertainty-models-resnet-packed-resnet101"]], "torch_uncertainty.models.resnet.packed_resnet152": [[19, "torch-uncertainty-models-resnet-packed-resnet152"]], "torch_uncertainty.models.resnet.packed_resnet18": [[20, "torch-uncertainty-models-resnet-packed-resnet18"]], "torch_uncertainty.models.resnet.packed_resnet34": [[21, "torch-uncertainty-models-resnet-packed-resnet34"]], "torch_uncertainty.models.resnet.packed_resnet50": [[22, "torch-uncertainty-models-resnet-packed-resnet50"]], "torch_uncertainty.models.resnet.resnet101": [[23, "torch-uncertainty-models-resnet-resnet101"]], "torch_uncertainty.models.resnet.resnet152": [[24, "torch-uncertainty-models-resnet-resnet152"]], "torch_uncertainty.models.resnet.resnet18": [[25, "torch-uncertainty-models-resnet-resnet18"]], "torch_uncertainty.models.resnet.resnet34": [[26, "torch-uncertainty-models-resnet-resnet34"]], "torch_uncertainty.models.resnet.resnet50": [[27, "torch-uncertainty-models-resnet-resnet50"]], "Torch Uncertainty": [[28, "torch-uncertainty"]], "Contents:": [[28, null]], "Indices and tables": [[28, "indices-and-tables"]], "Installation": [[29, "installation"]], "From PyPI": [[29, "from-pypi"]], "From source": [[29, "from-source"]], "Quickstart": [[30, "quickstart"]], "Using the CLI tool": [[30, "using-the-cli-tool"]], "Procedure": [[30, "procedure"]], "Exemple": [[30, "exemple"]], "Using your own trainer": [[30, "using-your-own-trainer"]]}, "indexentries": {"packedresnet (class in torch_uncertainty.baselines)": [[4, "torch_uncertainty.baselines.PackedResNet"]], "add_model_specific_args() (torch_uncertainty.baselines.packedresnet static method)": [[4, "torch_uncertainty.baselines.PackedResNet.add_model_specific_args"]], "packedconv2d (class in torch_uncertainty.layers)": [[5, "torch_uncertainty.layers.PackedConv2d"]], "packedlinear (class in torch_uncertainty.layers)": [[6, "torch_uncertainty.layers.PackedLinear"]], "brierscore (class in torch_uncertainty.metrics)": [[7, "torch_uncertainty.metrics.BrierScore"]], "compute() (torch_uncertainty.metrics.brierscore method)": [[7, "torch_uncertainty.metrics.BrierScore.compute"]], "update() (torch_uncertainty.metrics.brierscore method)": [[7, "torch_uncertainty.metrics.BrierScore.update"]], "disagreement (class in torch_uncertainty.metrics)": [[8, "torch_uncertainty.metrics.Disagreement"]], "compute() (torch_uncertainty.metrics.disagreement method)": [[8, "torch_uncertainty.metrics.Disagreement.compute"]], "update() (torch_uncertainty.metrics.disagreement method)": [[8, "torch_uncertainty.metrics.Disagreement.update"]], "entropy (class in torch_uncertainty.metrics)": [[9, "torch_uncertainty.metrics.Entropy"]], "compute() (torch_uncertainty.metrics.entropy method)": [[9, "torch_uncertainty.metrics.Entropy.compute"]], "update() (torch_uncertainty.metrics.entropy method)": [[9, "torch_uncertainty.metrics.Entropy.update"]], "jensenshannondivergence (class in torch_uncertainty.metrics)": [[10, "torch_uncertainty.metrics.JensenShannonDivergence"]], "compute() (torch_uncertainty.metrics.jensenshannondivergence method)": [[10, "torch_uncertainty.metrics.JensenShannonDivergence.compute"]], "update() (torch_uncertainty.metrics.jensenshannondivergence method)": [[10, "torch_uncertainty.metrics.JensenShannonDivergence.update"]], "mutualinformation (class in torch_uncertainty.metrics)": [[11, "torch_uncertainty.metrics.MutualInformation"]], "compute() (torch_uncertainty.metrics.mutualinformation method)": [[11, "torch_uncertainty.metrics.MutualInformation.compute"]], "update() (torch_uncertainty.metrics.mutualinformation method)": [[11, "torch_uncertainty.metrics.MutualInformation.update"]], "negativeloglikelihood (class in torch_uncertainty.metrics)": [[12, "torch_uncertainty.metrics.NegativeLogLikelihood"]], "compute() (torch_uncertainty.metrics.negativeloglikelihood method)": [[12, "torch_uncertainty.metrics.NegativeLogLikelihood.compute"]], "update() (torch_uncertainty.metrics.negativeloglikelihood method)": [[12, "torch_uncertainty.metrics.NegativeLogLikelihood.update"]], "masked_resnet101() (in module torch_uncertainty.models.resnet)": [[13, "torch_uncertainty.models.resnet.masked_resnet101"]], "masked_resnet152() (in module torch_uncertainty.models.resnet)": [[14, "torch_uncertainty.models.resnet.masked_resnet152"]], "masked_resnet18() (in module torch_uncertainty.models.resnet)": [[15, "torch_uncertainty.models.resnet.masked_resnet18"]], "masked_resnet34() (in module torch_uncertainty.models.resnet)": [[16, "torch_uncertainty.models.resnet.masked_resnet34"]], "masked_resnet50() (in module torch_uncertainty.models.resnet)": [[17, "torch_uncertainty.models.resnet.masked_resnet50"]], "packed_resnet101() (in module torch_uncertainty.models.resnet)": [[18, "torch_uncertainty.models.resnet.packed_resnet101"]], "packed_resnet152() (in module torch_uncertainty.models.resnet)": [[19, "torch_uncertainty.models.resnet.packed_resnet152"]], "packed_resnet18() (in module torch_uncertainty.models.resnet)": [[20, "torch_uncertainty.models.resnet.packed_resnet18"]], "packed_resnet34() (in module torch_uncertainty.models.resnet)": [[21, "torch_uncertainty.models.resnet.packed_resnet34"]], "packed_resnet50() (in module torch_uncertainty.models.resnet)": [[22, "torch_uncertainty.models.resnet.packed_resnet50"]], "resnet101() (in module torch_uncertainty.models.resnet)": [[23, "torch_uncertainty.models.resnet.resnet101"]], "resnet152() (in module torch_uncertainty.models.resnet)": [[24, "torch_uncertainty.models.resnet.resnet152"]], "resnet18() (in module torch_uncertainty.models.resnet)": [[25, "torch_uncertainty.models.resnet.resnet18"]], "resnet34() (in module torch_uncertainty.models.resnet)": [[26, "torch_uncertainty.models.resnet.resnet34"]], "resnet50() (in module torch_uncertainty.models.resnet)": [[27, "torch_uncertainty.models.resnet.resnet50"]]}})