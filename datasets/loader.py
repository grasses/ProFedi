#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/07/01, ZJUICSR'


import os
import numpy as np
import os.path as osp
from copy import deepcopy
from torchvision import datasets, transforms
from utils import helper
from datasets.base import BaseDataLoader, FediDataLoader
helper.set_seed(helper.get_args())


# This class provide default datasets config
class FediLoader(BaseDataLoader):
    def __init__(self, data_root, dataset="CIFAR10", **kwargs):
        self.dataset = dataset
        self.support = ["CIFAR10", "CIFAR100", "ImageNet32x", "Caltech256", "CUBS200", "SVHN", "MNIST", "FashionMNIST"]
        if self.dataset not in self.support:
            raise ValueError(f"-> System don't support {dataset}!!!")
        super(FediLoader, self).__init__(data_root, self.dataset, **kwargs)

        self.data_path = osp.join(data_root, dataset)
        if not osp.exists(self.data_path):
            os.makedirs(self.data_path)

        self._train = True
        self.train_loader = None
        self.test_loader = None
        self.train_transforms = None
        self.test_transforms = None
        self.params = self.__check_params__()
        self.get_transforms()
        self.__dump_params__()

    @staticmethod
    def __config__(dataset):
        params = {}
        params["name"] = dataset
        params["batch_size"] = 250
        params["size"] = FediLoader.get_size(dataset)
        params["shape"] = (3, params["size"][0], params["size"][1])
        params["mean"], params["std"] = FediLoader.get_mean_std(dataset)
        params["bounds"] = FediLoader.get_bounds(dataset)
        params["num_classes"] = FediLoader.get_num_classes(dataset)
        params["data_path"] = osp.join("datasets/data", dataset)
        params["labels"] = FediLoader.get_labels(dataset)
        return params

    def __dump_params__(self):
        keys = ["name", "batch_size", "size", "mean", "std", "bounds", "data_path"]
        print("\n-> dump datasets params:")
        data = {}
        for k in keys:
            data[k] = getattr(self, k)
        print(data)
        print()

    def __call__(self):
        if self._train:
            if self.train_loader is not None:
                for x, y in self.train_loader:
                    yield x, y
        else:
            if self.test_loader is not None:
                for x, y in self.test_loader:
                    yield x, y

    @staticmethod
    def get_num_classes(dataset):
        dnum = {
            "MNIST": 10,
            "FashionMNIST": 10,
            "CIFAR10": 10,
            "CIFAR100": 100,
            "ImageNet32x": 1000,
            "SVHN": 10,
            "CUBS200": 200,
            "Caltech256": 256
        }
        return dnum[dataset]

    @staticmethod
    def get_mean_std(dataset):
        attribute = {
            "MNIST": [(0.1307), (0.3081)],
            "FashionMNIST": [(0.1307), (0.3081)],
            "SVHN": [(0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)],
            "CIFAR": [(0.43768206, 0.44376972, 0.47280434), (0.19803014, 0.20101564, 0.19703615)],
            "CUBS200": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
            "Caltech256": [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
            "ImageNet32x": [(0.43768206, 0.44376972, 0.47280434), (0.19803014, 0.20101564, 0.19703615)]
        }
        attribute["CIFAR10"] = deepcopy(attribute["CIFAR"])
        attribute["CIFAR100"] = deepcopy(attribute["CIFAR"])
        return attribute[dataset]

    @staticmethod
    def get_size(dataset):
        attribute = {
            "MNIST": [32, 32],
            "SVHN": [32, 32],
            "CIFAR": [32, 32],
            "CUBS200": [128, 128],
            "Caltech256": [256, 256],
            "ImageNet32x": [32, 32],
        }
        attribute["FashionMNIST"] = deepcopy(attribute["MNIST"])
        attribute["CIFAR10"] = deepcopy(attribute["CIFAR"])
        attribute["CIFAR100"] = deepcopy(attribute["CIFAR"])
        return attribute[dataset]

    @staticmethod
    def get_bounds(dataset):
        mean, std = FediLoader.get_mean_std(dataset)
        bounds = [-1, 1]
        if type(mean) == type(()):
            c = len(mean)
            _min = (np.zeros([c]) - np.array(mean)) / np.array([std])
            _max = (np.ones([c]) - np.array(mean)) / np.array([std])
            bounds = [np.min(_min).item(), np.max(_max).item()]
        elif type(mean) == float:
            bounds = [(0.0 - mean) / std, (1.0 - mean) / std]
        return bounds

    def is_train(self, train=True):
        self._train = bool(train)

    def set_transforms(self, train_transforms, test_transforms):
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

    def get_transforms(self):
        if (self.train_transforms is not None) \
                and (self.test_transforms is not None):
            return self.train_transforms, self.test_transforms
        attribute = {
            "MNIST": [
                transforms.Compose([
                    transforms.Resize(self.size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ]), transforms.Compose([
                    transforms.Resize(self.size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std)
                ])
            ],
            "SVHN": [
                transforms.Compose([
                    transforms.RandomResizedCrop(self.size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]),
                transforms.Compose([
                    transforms.CenterCrop(self.size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ])
            ],
            "CIFAR": [
                transforms.Compose([
                    transforms.RandomCrop(self.size, padding=2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]),
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ])
            ],
            "CUBS200": [
                transforms.Compose([
                    transforms.CenterCrop(128),
                    transforms.Resize(self.size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]),
                transforms.Compose([
                    transforms.CenterCrop(128),
                    transforms.Resize(self.size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ])
            ],
            "Caltech256": [
                transforms.Compose([
                    transforms.CenterCrop(128),
                    transforms.Resize(self.size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]),
                transforms.Compose([
                    transforms.CenterCrop(128),
                    transforms.Resize(self.size),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ])
            ],
            "ImageNet32x": [
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]),
                transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ])
            ]
        }
        attribute["FashionMNIST"] = deepcopy(attribute["MNIST"])
        attribute["CIFAR10"] = deepcopy(attribute["CIFAR"])
        attribute["CIFAR100"] = deepcopy(attribute["CIFAR"])
        self.train_transforms, self.test_transforms = attribute[self.dataset]
        return attribute[self.dataset]

    @staticmethod
    def get_labels(dataset):
        dst = dataset
        attribute = {
            "CIFAR10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
            "CIFAR100": ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"],
            "MNIST": ["0", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
            "FashionMNIST": ["T-shirt", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "ImageNet32x": ['tench, Tinca tinca', 'goldfish, Carassius auratus', 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias', 'tiger shark, Galeocerdo cuvieri', 'hammerhead, hammerhead shark', 'electric ray, crampfish, numbfish, torpedo', 'stingray', 'cock', 'hen', 'ostrich, Struthio camelus', 'brambling, Fringilla montifringilla', 'goldfinch, Carduelis carduelis', 'house finch, linnet, Carpodacus mexicanus', 'junco, snowbird', 'indigo bunting, indigo finch, indigo bird, Passerina cyanea', 'robin, American robin, Turdus migratorius', 'bulbul', 'jay', 'magpie', 'chickadee', 'water ouzel, dipper', 'kite', 'bald eagle, American eagle, Haliaeetus leucocephalus', 'vulture', 'great grey owl, great gray owl, Strix nebulosa', 'European fire salamander, Salamandra salamandra', 'common newt, Triturus vulgaris', 'eft', 'spotted salamander, Ambystoma maculatum', 'axolotl, mud puppy, Ambystoma mexicanum', 'bullfrog, Rana catesbeiana', 'tree frog, tree-frog', 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui', 'loggerhead, loggerhead turtle, Caretta caretta', 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea', 'mud turtle', 'terrapin', 'box turtle, box tortoise', 'banded gecko', 'common iguana, iguana, Iguana iguana', 'American chameleon, anole, Anolis carolinensis', 'whiptail, whiptail lizard', 'agama', 'frilled lizard, Chlamydosaurus kingi', 'alligator lizard', 'Gila monster, Heloderma suspectum', 'green lizard, Lacerta viridis', 'African chameleon, Chamaeleo chamaeleon', 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis', 'African crocodile, Nile crocodile, Crocodylus niloticus', 'American alligator, Alligator mississipiensis', 'triceratops', 'thunder snake, worm snake, Carphophis amoenus', 'ringneck snake, ring-necked snake, ring snake', 'hognose snake, puff adder, sand viper', 'green snake, grass snake', 'king snake, kingsnake', 'garter snake, grass snake', 'water snake', 'vine snake', 'night snake, Hypsiglena torquata', 'boa constrictor, Constrictor constrictor', 'rock python, rock snake, Python sebae', 'Indian cobra, Naja naja', 'green mamba', 'sea snake', 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus', 'diamondback, diamondback rattlesnake, Crotalus adamanteus', 'sidewinder, horned rattlesnake, Crotalus cerastes', 'trilobite', 'harvestman, daddy longlegs, Phalangium opilio', 'scorpion', 'black and gold garden spider, Argiope aurantia', 'barn spider, Araneus cavaticus', 'garden spider, Aranea diademata', 'black widow, Latrodectus mactans', 'tarantula', 'wolf spider, hunting spider', 'tick', 'centipede', 'black grouse', 'ptarmigan', 'ruffed grouse, partridge, Bonasa umbellus', 'prairie chicken, prairie grouse, prairie fowl', 'peacock', 'quail', 'partridge', 'African grey, African gray, Psittacus erithacus', 'macaw', 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita', 'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird', 'jacamar', 'toucan', 'drake', 'red-breasted merganser, Mergus serrator', 'goose', 'black swan, Cygnus atratus', 'tusker', 'echidna, spiny anteater, anteater', 'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus', 'wallaby, brush kangaroo', 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 'wombat', 'jellyfish', 'sea anemone, anemone', 'brain coral', 'flatworm, platyhelminth', 'nematode, nematode worm, roundworm', 'conch', 'snail', 'slug', 'sea slug, nudibranch', 'chiton, coat-of-mail shell, sea cradle, polyplacophore', 'chambered nautilus, pearly nautilus, nautilus', 'Dungeness crab, Cancer magister', 'rock crab, Cancer irroratus', 'fiddler crab', 'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica', 'American lobster, Northern lobster, Maine lobster, Homarus americanus', 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish', 'crayfish, crawfish, crawdad, crawdaddy', 'hermit crab', 'isopod', 'white stork, Ciconia ciconia', 'black stork, Ciconia nigra', 'spoonbill', 'flamingo', 'little blue heron, Egretta caerulea', 'American egret, great white heron, Egretta albus', 'bittern', 'crane', 'limpkin, Aramus pictus', 'European gallinule, Porphyrio porphyrio', 'American coot, marsh hen, mud hen, water hen, Fulica americana', 'bustard', 'ruddy turnstone, Arenaria interpres', 'red-backed sandpiper, dunlin, Erolia alpina', 'redshank, Tringa totanus', 'dowitcher', 'oystercatcher, oyster catcher', 'pelican', 'king penguin, Aptenodytes patagonica', 'albatross, mollymawk', 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus', 'killer whale, killer, orca, grampus, sea wolf, Orcinus orca', 'dugong, Dugong dugon', 'sea lion', 'Chihuahua', 'Japanese spaniel', 'Maltese dog, Maltese terrier, Maltese', 'Pekinese, Pekingese, Peke', 'Shih-Tzu', 'Blenheim spaniel', 'papillon', 'toy terrier', 'Rhodesian ridgeback', 'Afghan hound, Afghan', 'basset, basset hound', 'beagle', 'bloodhound, sleuthhound', 'bluetick', 'black-and-tan coonhound', 'Walker hound, Walker foxhound', 'English foxhound', 'redbone', 'borzoi, Russian wolfhound', 'Irish wolfhound', 'Italian greyhound', 'whippet', 'Ibizan hound, Ibizan Podenco', 'Norwegian elkhound, elkhound', 'otterhound, otter hound', 'Saluki, gazelle hound', 'Scottish deerhound, deerhound', 'Weimaraner', 'Staffordshire bullterrier, Staffordshire bull terrier', 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier', 'Bedlington terrier', 'Border terrier', 'Kerry blue terrier', 'Irish terrier', 'Norfolk terrier', 'Norwich terrier', 'Yorkshire terrier', 'wire-haired fox terrier', 'Lakeland terrier', 'Sealyham terrier, Sealyham', 'Airedale, Airedale terrier', 'cairn, cairn terrier', 'Australian terrier', 'Dandie Dinmont, Dandie Dinmont terrier', 'Boston bull, Boston terrier', 'miniature schnauzer', 'giant schnauzer', 'standard schnauzer', 'Scotch terrier, Scottish terrier, Scottie', 'Tibetan terrier, chrysanthemum dog', 'silky terrier, Sydney silky', 'soft-coated wheaten terrier', 'West Highland white terrier', 'Lhasa, Lhasa apso', 'flat-coated retriever', 'curly-coated retriever', 'golden retriever', 'Labrador retriever', 'Chesapeake Bay retriever', 'German short-haired pointer', 'vizsla, Hungarian pointer', 'English setter', 'Irish setter, red setter', 'Gordon setter', 'Brittany spaniel', 'clumber, clumber spaniel', 'English springer, English springer spaniel', 'Welsh springer spaniel', 'cocker spaniel, English cocker spaniel, cocker', 'Sussex spaniel', 'Irish water spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard', 'kelpie', 'komondor', 'Old English sheepdog, bobtail', 'Shetland sheepdog, Shetland sheep dog, Shetland', 'collie', 'Border collie', 'Bouvier des Flandres, Bouviers des Flandres', 'Rottweiler', 'German shepherd, German shepherd dog, German police dog, alsatian', 'Doberman, Doberman pinscher', 'miniature pinscher', 'Greater Swiss Mountain dog', 'Bernese mountain dog', 'Appenzeller', 'EntleBucher', 'boxer', 'bull mastiff', 'Tibetan mastiff', 'French bulldog', 'Great Dane', 'Saint Bernard, St Bernard', 'Eskimo dog, husky', 'malamute, malemute, Alaskan malamute', 'Siberian husky', 'dalmatian, coach dog, carriage dog', 'affenpinscher, monkey pinscher, monkey dog', 'basenji', 'pug, pug-dog', 'Leonberg', 'Newfoundland, Newfoundland dog', 'Great Pyrenees', 'Samoyed, Samoyede', 'Pomeranian', 'chow, chow chow', 'keeshond', 'Brabancon griffon', 'Pembroke, Pembroke Welsh corgi', 'Cardigan, Cardigan Welsh corgi', 'toy poodle', 'miniature poodle', 'standard poodle', 'Mexican hairless', 'timber wolf, grey wolf, gray wolf, Canis lupus', 'white wolf, Arctic wolf, Canis lupus tundrarum', 'red wolf, maned wolf, Canis rufus, Canis niger', 'coyote, prairie wolf, brush wolf, Canis latrans', 'dingo, warrigal, warragal, Canis dingo', 'dhole, Cuon alpinus', 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus', 'hyena, hyaena', 'red fox, Vulpes vulpes', 'kit fox, Vulpes macrotis', 'Arctic fox, white fox, Alopex lagopus', 'grey fox, gray fox, Urocyon cinereoargenteus', 'tabby, tabby cat', 'tiger cat', 'Persian cat', 'Siamese cat, Siamese', 'Egyptian cat', 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor', 'lynx, catamount', 'leopard, Panthera pardus', 'snow leopard, ounce, Panthera uncia', 'jaguar, panther, Panthera onca, Felis onca', 'lion, king of beasts, Panthera leo', 'tiger, Panthera tigris', 'cheetah, chetah, Acinonyx jubatus', 'brown bear, bruin, Ursus arctos', 'American black bear, black bear, Ursus americanus, Euarctos americanus', 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'sloth bear, Melursus ursinus, Ursus ursinus', 'mongoose', 'meerkat, mierkat', 'tiger beetle', 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'ground beetle, carabid beetle', 'long-horned beetle, longicorn, longicorn beetle', 'leaf beetle, chrysomelid', 'dung beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant, emmet, pismire', 'grasshopper, hopper', 'cricket', 'walking stick, walkingstick, stick insect', 'cockroach, roach', 'mantis, mantid', 'cicada, cicala', 'leafhopper', 'lacewing, lacewing fly', "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk", 'damselfly', 'admiral', 'ringlet, ringlet butterfly', 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus', 'cabbage butterfly', 'sulphur butterfly, sulfur butterfly', 'lycaenid, lycaenid butterfly', 'starfish, sea star', 'sea urchin', 'sea cucumber, holothurian', 'wood rabbit, cottontail, cottontail rabbit', 'hare', 'Angora, Angora rabbit', 'hamster', 'porcupine, hedgehog', 'fox squirrel, eastern fox squirrel, Sciurus niger', 'marmot', 'beaver', 'guinea pig, Cavia cobaya', 'sorrel', 'zebra', 'hog, pig, grunter, squealer, Sus scrofa', 'wild boar, boar, Sus scrofa', 'warthog', 'hippopotamus, hippo, river horse, Hippopotamus amphibius', 'ox', 'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis', 'bison', 'ram, tup', 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis', 'ibex, Capra ibex', 'hartebeest', 'impala, Aepyceros melampus', 'gazelle', 'Arabian camel, dromedary, Camelus dromedarius', 'llama', 'weasel', 'mink', 'polecat, fitch, foulmart, foumart, Mustela putorius', 'black-footed ferret, ferret, Mustela nigripes', 'otter', 'skunk, polecat, wood pussy', 'badger', 'armadillo', 'three-toed sloth, ai, Bradypus tridactylus', 'orangutan, orang, orangutang, Pongo pygmaeus', 'gorilla, Gorilla gorilla', 'chimpanzee, chimp, Pan troglodytes', 'gibbon, Hylobates lar', 'siamang, Hylobates syndactylus, Symphalangus syndactylus', 'guenon, guenon monkey', 'patas, hussar monkey, Erythrocebus patas', 'baboon', 'macaque', 'langur', 'colobus, colobus monkey', 'proboscis monkey, Nasalis larvatus', 'marmoset', 'capuchin, ringtail, Cebus capucinus', 'howler monkey, howler', 'titi, titi monkey', 'spider monkey, Ateles geoffroyi', 'squirrel monkey, Saimiri sciureus', 'Madagascar cat, ring-tailed lemur, Lemur catta', 'indri, indris, Indri indri, Indri brevicaudatus', 'Indian elephant, Elephas maximus', 'African elephant, Loxodonta africana', 'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens', 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca', 'barracouta, snoek', 'eel', 'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch', 'rock beauty, Holocanthus tricolor', 'anemone fish', 'sturgeon', 'gar, garfish, garpike, billfish, Lepisosteus osseus', 'lionfish', 'puffer, pufferfish, blowfish, globefish', 'abacus', 'abaya', "academic gown, academic robe, judge's robe", 'accordion, piano accordion, squeeze box', 'acoustic guitar', 'aircraft carrier, carrier, flattop, attack aircraft carrier', 'airliner', 'airship, dirigible', 'altar', 'ambulance', 'amphibian, amphibious vehicle', 'analog clock', 'apiary, bee house', 'apron', 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin', 'assault rifle, assault gun', 'backpack, back pack, knapsack, packsack, rucksack, haversack', 'bakery, bakeshop, bakehouse', 'balance beam, beam', 'balloon', 'ballpoint, ballpoint pen, ballpen, Biro', 'Band Aid', 'banjo', 'bannister, banister, balustrade, balusters, handrail', 'barbell', 'barber chair', 'barbershop', 'barn', 'barometer', 'barrel, cask', 'barrow, garden cart, lawn cart, wheelbarrow', 'baseball', 'basketball', 'bassinet', 'bassoon', 'bathing cap, swimming cap', 'bath towel', 'bathtub, bathing tub, bath, tub', 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon', 'beacon, lighthouse, beacon light, pharos', 'beaker', 'bearskin, busby, shako', 'beer bottle', 'beer glass', 'bell cote, bell cot', 'bib', 'bicycle-built-for-two, tandem bicycle, tandem', 'bikini, two-piece', 'binder, ring-binder', 'binoculars, field glasses, opera glasses', 'birdhouse', 'boathouse', 'bobsled, bobsleigh, bob', 'bolo tie, bolo, bola tie, bola', 'bonnet, poke bonnet', 'bookcase', 'bookshop, bookstore, bookstall', 'bottlecap', 'bow', 'bow tie, bow-tie, bowtie', 'brass, memorial tablet, plaque', 'brassiere, bra, bandeau', 'breakwater, groin, groyne, mole, bulwark, seawall, jetty', 'breastplate, aegis, egis', 'broom', 'bucket, pail', 'buckle', 'bulletproof vest', 'bullet train, bullet', 'butcher shop, meat market', 'cab, hack, taxi, taxicab', 'caldron, cauldron', 'candle, taper, wax light', 'cannon', 'canoe', 'can opener, tin opener', 'cardigan', 'car mirror', 'carousel, carrousel, merry-go-round, roundabout, whirligig', "carpenter's kit, tool kit", 'carton', 'car wheel', 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM', 'cassette', 'cassette player', 'castle', 'catamaran', 'CD player', 'cello, violoncello', 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'chain', 'chainlink fence', 'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour', 'chain saw, chainsaw', 'chest', 'chiffonier, commode', 'chime, bell, gong', 'china cabinet, china closet', 'Christmas stocking', 'church, church building', 'cinema, movie theater, movie theatre, movie house, picture palace', 'cleaver, meat cleaver, chopper', 'cliff dwelling', 'cloak', 'clog, geta, patten, sabot', 'cocktail shaker', 'coffee mug', 'coffeepot', 'coil, spiral, volute, whorl, helix', 'combination lock', 'computer keyboard, keypad', 'confectionery, confectionary, candy store', 'container ship, containership, container vessel', 'convertible', 'corkscrew, bottle screw', 'cornet, horn, trumpet, trump', 'cowboy boot', 'cowboy hat, ten-gallon hat', 'cradle', 'crane', 'crash helmet', 'crate', 'crib, cot', 'Crock Pot', 'croquet ball', 'crutch', 'cuirass', 'dam, dike, dyke', 'desk', 'desktop computer', 'dial telephone, dial phone', 'diaper, nappy, napkin', 'digital clock', 'digital watch', 'dining table, board', 'dishrag, dishcloth', 'dishwasher, dish washer, dishwashing machine', 'disk brake, disc brake', 'dock, dockage, docking facility', 'dogsled, dog sled, dog sleigh', 'dome', 'doormat, welcome mat', 'drilling platform, offshore rig', 'drum, membranophone, tympan', 'drumstick', 'dumbbell', 'Dutch oven', 'electric fan, blower', 'electric guitar', 'electric locomotive', 'entertainment center', 'envelope', 'espresso maker', 'face powder', 'feather boa, boa', 'file, file cabinet, filing cabinet', 'fireboat', 'fire engine, fire truck', 'fire screen, fireguard', 'flagpole, flagstaff', 'flute, transverse flute', 'folding chair', 'football helmet', 'forklift', 'fountain', 'fountain pen', 'four-poster', 'freight car', 'French horn, horn', 'frying pan, frypan, skillet', 'fur coat', 'garbage truck, dustcart', 'gasmask, respirator, gas helmet', 'gas pump, gasoline pump, petrol pump, island dispenser', 'goblet', 'go-kart', 'golf ball', 'golfcart, golf cart', 'gondola', 'gong, tam-tam', 'gown', 'grand piano, grand', 'greenhouse, nursery, glasshouse', 'grille, radiator grille', 'grocery store, grocery, food market, market', 'guillotine', 'hair slide', 'hair spray', 'half track', 'hammer', 'hamper', 'hand blower, blow dryer, blow drier, hair dryer, hair drier', 'hand-held computer, hand-held microcomputer', 'handkerchief, hankie, hanky, hankey', 'hard disc, hard disk, fixed disk', 'harmonica, mouth organ, harp, mouth harp', 'harp', 'harvester, reaper', 'hatchet', 'holster', 'home theater, home theatre', 'honeycomb', 'hook, claw', 'hoopskirt, crinoline', 'horizontal bar, high bar', 'horse cart, horse-cart', 'hourglass', 'iPod', 'iron, smoothing iron', "jack-o'-lantern", 'jean, blue jean, denim', 'jeep, landrover', 'jersey, T-shirt, tee shirt', 'jigsaw puzzle', 'jinrikisha, ricksha, rickshaw', 'joystick', 'kimono', 'knee pad', 'knot', 'lab coat, laboratory coat', 'ladle', 'lampshade, lamp shade', 'laptop, laptop computer', 'lawn mower, mower', 'lens cap, lens cover', 'letter opener, paper knife, paperknife', 'library', 'lifeboat', 'lighter, light, igniter, ignitor', 'limousine, limo', 'liner, ocean liner', 'lipstick, lip rouge', 'Loafer', 'lotion', 'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system', "loupe, jeweler's loupe", 'lumbermill, sawmill', 'magnetic compass', 'mailbag, postbag', 'mailbox, letter box', 'maillot', 'maillot, tank suit', 'manhole cover', 'maraca', 'marimba, xylophone', 'mask', 'matchstick', 'maypole', 'maze, labyrinth', 'measuring cup', 'medicine chest, medicine cabinet', 'megalith, megalithic structure', 'microphone, mike', 'microwave, microwave oven', 'military uniform', 'milk can', 'minibus', 'miniskirt, mini', 'minivan', 'missile', 'mitten', 'mixing bowl', 'mobile home, manufactured home', 'Model T', 'modem', 'monastery', 'monitor', 'moped', 'mortar', 'mortarboard', 'mosque', 'mosquito net', 'motor scooter, scooter', 'mountain bike, all-terrain bike, off-roader', 'mountain tent', 'mouse, computer mouse', 'mousetrap', 'moving van', 'muzzle', 'nail', 'neck brace', 'necklace', 'nipple', 'notebook, notebook computer', 'obelisk', 'oboe, hautboy, hautbois', 'ocarina, sweet potato', 'odometer, hodometer, mileometer, milometer', 'oil filter', 'organ, pipe organ', 'oscilloscope, scope, cathode-ray oscilloscope, CRO', 'overskirt', 'oxcart', 'oxygen mask', 'packet', 'paddle, boat paddle', 'paddlewheel, paddle wheel', 'padlock', 'paintbrush', "pajama, pyjama, pj's, jammies", 'palace', 'panpipe, pandean pipe, syrinx', 'paper towel', 'parachute, chute', 'parallel bars, bars', 'park bench', 'parking meter', 'passenger car, coach, carriage', 'patio, terrace', 'pay-phone, pay-station', 'pedestal, plinth, footstall', 'pencil box, pencil case', 'pencil sharpener', 'perfume, essence', 'Petri dish', 'photocopier', 'pick, plectrum, plectron', 'pickelhaube', 'picket fence, paling', 'pickup, pickup truck', 'pier', 'piggy bank, penny bank', 'pill bottle', 'pillow', 'ping-pong ball', 'pinwheel', 'pirate, pirate ship', 'pitcher, ewer', "plane, carpenter's plane, woodworking plane", 'planetarium', 'plastic bag', 'plate rack', 'plow, plough', "plunger, plumber's helper", 'Polaroid camera, Polaroid Land camera', 'pole', 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria', 'poncho', 'pool table, billiard table, snooker table', 'pop bottle, soda bottle', 'pot, flowerpot', "potter's wheel", 'power drill', 'prayer rug, prayer mat', 'printer', 'prison, prison house', 'projectile, missile', 'projector', 'puck, hockey puck', 'punching bag, punch bag, punching ball, punchball', 'purse', 'quill, quill pen', 'quilt, comforter, comfort, puff', 'racer, race car, racing car', 'racket, racquet', 'radiator', 'radio, wireless', 'radio telescope, radio reflector', 'rain barrel', 'recreational vehicle, RV, R.V.', 'reel', 'reflex camera', 'refrigerator, icebox', 'remote control, remote', 'restaurant, eating house, eating place, eatery', 'revolver, six-gun, six-shooter', 'rifle', 'rocking chair, rocker', 'rotisserie', 'rubber eraser, rubber, pencil eraser', 'rugby ball', 'rule, ruler', 'running shoe', 'safe', 'safety pin', 'saltshaker, salt shaker', 'sandal', 'sarong', 'sax, saxophone', 'scabbard', 'scale, weighing machine', 'school bus', 'schooner', 'scoreboard', 'screen, CRT screen', 'screw', 'screwdriver', 'seat belt, seatbelt', 'sewing machine', 'shield, buckler', 'shoe shop, shoe-shop, shoe store', 'shoji', 'shopping basket', 'shopping cart', 'shovel', 'shower cap', 'shower curtain', 'ski', 'ski mask', 'sleeping bag', 'slide rule, slipstick', 'sliding door', 'slot, one-armed bandit', 'snorkel', 'snowmobile', 'snowplow, snowplough', 'soap dispenser', 'soccer ball', 'sock', 'solar dish, solar collector, solar furnace', 'sombrero', 'soup bowl', 'space bar', 'space heater', 'space shuttle', 'spatula', 'speedboat', "spider web, spider's web", 'spindle', 'sports car, sport car', 'spotlight, spot', 'stage', 'steam locomotive', 'steel arch bridge', 'steel drum', 'stethoscope', 'stole', 'stone wall', 'stopwatch, stop watch', 'stove', 'strainer', 'streetcar, tram, tramcar, trolley, trolley car', 'stretcher', 'studio couch, day bed', 'stupa, tope', 'submarine, pigboat, sub, U-boat', 'suit, suit of clothes', 'sundial', 'sunglass', 'sunglasses, dark glasses, shades', 'sunscreen, sunblock, sun blocker', 'suspension bridge', 'swab, swob, mop', 'sweatshirt', 'swimming trunks, bathing trunks', 'swing', 'switch, electric switch, electrical switch', 'syringe', 'table lamp', 'tank, army tank, armored combat vehicle, armoured combat vehicle', 'tape player', 'teapot', 'teddy, teddy bear', 'television, television system', 'tennis ball', 'thatch, thatched roof', 'theater curtain, theatre curtain', 'thimble', 'thresher, thrasher, threshing machine', 'throne', 'tile roof', 'toaster', 'tobacco shop, tobacconist shop, tobacconist', 'toilet seat', 'torch', 'totem pole', 'tow truck, tow car, wrecker', 'toyshop', 'tractor', 'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi', 'tray', 'trench coat', 'tricycle, trike, velocipede', 'trimaran', 'tripod', 'triumphal arch', 'trolleybus, trolley coach, trackless trolley', 'trombone', 'tub, vat', 'turnstile', 'typewriter keyboard', 'umbrella', 'unicycle, monocycle', 'upright, upright piano', 'vacuum, vacuum cleaner', 'vase', 'vault', 'velvet', 'vending machine', 'vestment', 'viaduct', 'violin, fiddle', 'volleyball', 'waffle iron', 'wall clock', 'wallet, billfold, notecase, pocketbook', 'wardrobe, closet, press', 'warplane, military plane', 'washbasin, handbasin, washbowl, lavabo, wash-hand basin', 'washer, automatic washer, washing machine', 'water bottle', 'water jug', 'water tower', 'whiskey jug', 'whistle', 'wig', 'window screen', 'window shade', 'Windsor tie', 'wine bottle', 'wing', 'wok', 'wooden spoon', 'wool, woolen, woollen', 'worm fence, snake fence, snake-rail fence, Virginia fence', 'wreck', 'yawl', 'yurt', 'web site, website, internet site, site', 'comic book', 'crossword puzzle, crossword', 'street sign', 'traffic light, traffic signal, stoplight', 'book jacket, dust cover, dust jacket, dust wrapper', 'menu', 'plate', 'guacamole', 'consomme', 'hot pot, hotpot', 'trifle', 'ice cream, icecream', 'ice lolly, lolly, lollipop, popsicle', 'French loaf', 'bagel, beigel', 'pretzel', 'cheeseburger', 'hotdog, hot dog, red hot', 'mashed potato', 'head cabbage', 'broccoli', 'cauliflower', 'zucchini, courgette', 'spaghetti squash', 'acorn squash', 'butternut squash', 'cucumber, cuke', 'artichoke, globe artichoke', 'bell pepper', 'cardoon', 'mushroom', 'Granny Smith', 'strawberry', 'orange', 'lemon', 'fig', 'pineapple, ananas', 'banana', 'jackfruit, jak, jack', 'custard apple', 'pomegranate', 'hay', 'carbonara', 'chocolate sauce, chocolate syrup', 'dough', 'meat loaf, meatloaf', 'pizza, pizza pie', 'potpie', 'burrito', 'red wine', 'espresso', 'cup', 'eggnog', 'alp', 'bubble', 'cliff, drop, drop-off', 'coral reef', 'geyser', 'lakeside, lakeshore', 'promontory, headland, head, foreland', 'sandbar, sand bar', 'seashore, coast, seacoast, sea-coast', 'valley, vale', 'volcano', 'ballplayer, baseball player', 'groom, bridegroom', 'scuba diver', 'rapeseed', 'daisy', "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum", 'corn', 'acorn', 'hip, rose hip, rosehip', 'buckeye, horse chestnut, conker', 'coral fungus', 'agaric', 'gyromitra', 'stinkhorn, carrion fungus', 'earthstar', 'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa', 'bolete', 'ear, spike, capitulum', 'toilet tissue, toilet paper, bathroom tissue'],
            "Caltech256": ["001.ak47", "002.american-flag", "003.backpack", "004.baseball-bat", "005.baseball-glove", "006.basketball-hoop", "007.bat", "008.bathtub", "009.bear", "010.beer-mug", "011.billiards", "012.binoculars", "013.birdbath", "014.blimp", "015.bonsai-101", "016.boom-box", "017.bowling-ball", "018.bowling-pin", "019.boxing-glove", "020.brain-101", "021.breadmaker", "022.buddha-101", "023.bulldozer", "024.butterfly", "025.cactus", "026.cake", "027.calculator", "028.camel", "029.cannon", "030.canoe", "031.car-tire", "032.cartman", "033.cd", "034.centipede", "035.cereal-box", "036.chandelier-101", "037.chess-board", "038.chimp", "039.chopsticks", "040.cockroach", "041.coffee-mug", "042.coffin", "043.coin", "044.comet", "045.computer-keyboard", "046.computer-monitor", "047.computer-mouse", "048.conch", "049.cormorant", "050.covered-wagon", "051.cowboy-hat", "052.crab-101", "053.desk-globe", "054.diamond-ring", "055.dice", "056.dog", "057.dolphin-101", "058.doorknob", "059.drinking-straw", "060.duck", "061.dumb-bell", "062.eiffel-tower", "063.electric-guitar-101", "064.elephant-101", "065.elk", "066.ewer-101", "067.eyeglasses", "068.fern", "069.fighter-jet", "070.fire-extinguisher", "071.fire-hydrant", "072.fire-truck", "073.fireworks", "074.flashlight", "075.floppy-disk", "076.football-helmet", "077.french-horn", "078.fried-egg", "079.frisbee", "080.frog", "081.frying-pan", "082.galaxy", "083.gas-pump", "084.giraffe", "085.goat", "086.golden-gate-bridge", "087.goldfish", "088.golf-ball", "089.goose", "090.gorilla", "091.grand-piano-101", "092.grapes", "093.grasshopper", "094.guitar-pick", "095.hamburger", "096.hammock", "097.harmonica", "098.harp", "099.harpsichord", "100.hawksbill-101", "101.head-phones", "102.helicopter-101", "103.hibiscus", "104.homer-simpson", "105.horse", "106.horseshoe-crab", "107.hot-air-balloon", "108.hot-dog", "109.hot-tub", "110.hourglass", "111.house-fly", "112.human-skeleton", "113.hummingbird", "114.ibis-101", "115.ice-cream-cone", "116.iguana", "117.ipod", "118.iris", "119.jesus-christ", "120.joy-stick", "121.kangaroo-101", "122.kayak", "123.ketch-101", "124.killer-whale", "125.knife", "126.ladder", "127.laptop-101", "128.lathe", "129.leopards-101", "130.license-plate", "131.lightbulb", "132.light-house", "133.lightning", "134.llama-101", "135.mailbox", "136.mandolin", "137.mars", "138.mattress", "139.megaphone", "140.menorah-101", "141.microscope", "142.microwave", "143.minaret", "144.minotaur", "145.motorbikes-101", "146.mountain-bike", "147.mushroom", "148.mussels", "149.necktie", "150.octopus", "151.ostrich", "152.owl", "153.palm-pilot", "154.palm-tree", "155.paperclip", "156.paper-shredder", "157.pci-card", "158.penguin", "159.people", "160.pez-dispenser", "161.photocopier", "162.picnic-table", "163.playing-card", "164.porcupine", "165.pram", "166.praying-mantis", "167.pyramid", "168.raccoon", "169.radio-telescope", "170.rainbow", "171.refrigerator", "172.revolver-101", "173.rifle", "174.rotary-phone", "175.roulette-wheel", "176.saddle", "177.saturn", "178.school-bus", "179.scorpion-101", "180.screwdriver", "181.segway", "182.self-propelled-lawn-mower", "183.sextant", "184.sheet-music", "185.skateboard", "186.skunk", "187.skyscraper", "188.smokestack", "189.snail", "190.snake", "191.sneaker", "192.snowmobile", "193.soccer-ball", "194.socks", "195.soda-can", "196.spaghetti", "197.speed-boat", "198.spider", "199.spoon", "200.stained-glass", "201.starfish-101", "202.steering-wheel", "203.stirrups", "204.sunflower-101", "205.superman", "206.sushi", "207.swan", "208.swiss-army-knife", "209.sword", "210.syringe", "211.tambourine", "212.teapot", "213.teddy-bear", "214.teepee", "215.telephone-box", "216.tennis-ball", "217.tennis-court", "218.tennis-racket", "219.theodolite", "220.toaster", "221.tomato", "222.tombstone", "223.top-hat", "224.touring-bike", "225.tower-pisa", "226.traffic-light", "227.treadmill", "228.triceratops", "229.tricycle", "230.trilobite-101", "231.tripod", "232.t-shirt", "233.tuning-fork", "234.tweezer", "235.umbrella-101", "236.unicorn", "237.vcr", "238.video-projector", "239.washing-machine", "240.watch-101", "241.waterfall", "242.watermelon", "243.welding-mask", "244.wheelbarrow", "245.windmill", "246.wine-bottle", "247.xylophone", "248.yarmulke", "249.yo-yo", "250.zebra", "251.airplanes-101", "252.car-side-101", "253.faces-easy-101", "254.greyhound", "255.tennis-shoes", "256.toad"],
        }
        attribute["SVHN"] = deepcopy(attribute["MNIST"])
        return attribute[dst]

    def get_loader(self, **kwargs):
        train_transforms, test_transforms = self.get_transforms()
        if self.dataset == "MNIST":
            self.train_loader, self.test_loader = [
                FediDataLoader(
                    dataset=datasets.MNIST(self.data_path, train=True, download=True, transform=train_transforms),
                    batch_size=self.batch_size,
                    shuffle=True, num_workers=4,
                    **kwargs
                ),
                FediDataLoader(
                    dataset=datasets.MNIST(self.data_path, train=False, download=True, transform=test_transforms),
                    batch_size=self.batch_size,
                    shuffle=False, num_workers=4,
                    **kwargs
                )]
        elif self.dataset == "FashionMNIST":
            self.train_loader, self.test_loader = [
                FediDataLoader(
                    dataset=datasets.FashionMNIST(self.data_path, train=True, download=True, transform=train_transforms),
                    batch_size=self.batch_size,
                    shuffle=True, num_workers=4,
                    **kwargs
                ),
                FediDataLoader(
                    dataset=datasets.FashionMNIST(self.data_path, train=False, download=True, transform=test_transforms),
                    batch_size=self.batch_size,
                    shuffle=False, num_workers=4,
                    **kwargs
                )]
        elif self.dataset == "CIFAR10":
            self.train_loader, self.test_loader = [
                FediDataLoader(
                    dataset=datasets.CIFAR10(self.data_path, train=True, download=True, transform=train_transforms),
                    batch_size=self.batch_size,
                    shuffle=True, num_workers=4,
                    **kwargs
                ),
                FediDataLoader(
                    dataset=datasets.CIFAR10(self.data_path, train=False, download=True, transform=test_transforms),
                    batch_size=self.batch_size,
                    shuffle=False, num_workers=4,
                    **kwargs
                )
            ]
        elif self.dataset == "CIFAR100":
            self.train_loader, self.test_loader = [
                FediDataLoader(
                    dataset=datasets.CIFAR100(self.data_path, train=True, download=True, transform=train_transforms),
                    batch_size=self.batch_size,
                    shuffle=True, num_workers=4,
                    **kwargs
                ),
                FediDataLoader(
                    dataset=datasets.CIFAR100(self.data_path, train=False, download=True, transform=test_transforms),
                    batch_size=self.batch_size,
                    shuffle=False, num_workers=4,
                    **kwargs
                )
            ]
        elif self.dataset == "SVHN":
            self.train_loader, self.test_loader = [
                FediDataLoader(
                    dataset=datasets.SVHN(self.data_path, split='train', download=True, transform=train_transforms),
                    batch_size=self.batch_size,
                    shuffle=True, num_workers=4,
                    **kwargs
                ),
                FediDataLoader(
                    dataset=datasets.SVHN(self.data_path, split='test', download=True, transform=test_transforms),
                    batch_size=self.batch_size,
                    shuffle=False, num_workers=4,
                    **kwargs
                )
            ]
        elif self.dataset == "ImageNet32x":
            from datasets.imagenet32x import ImageNet32x
            self.train_loader, self.test_loader = [
                FediDataLoader(
                    dataset=ImageNet32x(root=self.data_path, train=True, transform=train_transforms),
                    batch_size=self.batch_size,
                    shuffle=True, num_workers=4,
                    **kwargs
                ),
                FediDataLoader(
                    dataset=ImageNet32x(root=self.data_path, train=False, transform=test_transforms),
                    batch_size=self.batch_size,
                    shuffle=False, num_workers=4,
                    **kwargs
                )
            ]

        else:
            raise NotImplementedError(f"-> Can't find {self.dataset} implementation!!")

        self.train_loader.set_params(self.params)
        self.train_loader.set_params({
            "transforms": self.train_transforms
        })
        self.test_loader.set_params(self.params)
        self.test_loader.set_params({
            "transforms": self.test_transforms
        })
        return self.train_loader, self.test_loader















