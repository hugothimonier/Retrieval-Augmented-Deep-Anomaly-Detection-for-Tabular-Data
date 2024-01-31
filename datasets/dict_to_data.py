from datasets.base import BaseDataset
from datasets.arrhythmia import ArrhythmiaDataset
from datasets.thyroid import ThyroidDataset
from datasets.separable import SeparableDataset
from datasets.annthyroid import AnnthyroidDataset
from datasets.abalone import AbaloneDataset
from datasets.breastw import BreastWDataset
from datasets.forestcoverad import ForestCoverADDataset
from datasets.glass import GlassDataset
from datasets.ionosphere import IonosphereDataset
from datasets.cardio import CardioDataset
from datasets.letter import LetterDataset
from datasets.lympho import LymphoDataset
from datasets.mammography import MammographyDataset
from datasets.mnistad import MnistADDataset
from datasets.musk import MuskDataset
from datasets.optdigits import OptdigitsDataset
from datasets.pendigits import PendigitsDataset
from datasets.pima import PimaDataset
from datasets.satellite import SatelliteDataset
from datasets.satimage import SatimageDataset
from datasets.shuttle import ShuttleDataset
from datasets.speech import SpeechDataset
from datasets.vertebral import VertebralDataset
from datasets.vowels import VowelsDataset
from datasets.wbc import WbcDataset
from datasets.wine import WineDataset
from datasets.seismic import SeismicDataset
from datasets.mulcross import MulcrossDataset
from datasets.ecoli import EcoliDataset
from datasets.fraud import FraudDataset
from datasets.campaign import CampaignDataset
from datasets.backdoor import BackdoorDataset

DATASET_NAME_TO_DATASET_MAP = {
    'arrhythmia': ArrhythmiaDataset,
    'thyroid': ThyroidDataset,
    'separable': SeparableDataset,
    'annthyroid': AnnthyroidDataset,
    'abalone': AbaloneDataset,
    'breastw': BreastWDataset,
    'forest': ForestCoverADDataset,
    'glass': GlassDataset,
    'ionosphere': IonosphereDataset,
    'letter': LetterDataset,
    'lympho': LymphoDataset,
    'mammography': MammographyDataset,
    'mnist': MnistADDataset,
    'musk': MuskDataset,
    'optdigits': OptdigitsDataset,
    'pendigits': PendigitsDataset,
    'pima': PimaDataset,
    'satellite': SatelliteDataset,
    'satimage': SatimageDataset,
    'shuttle': ShuttleDataset,
    'speech': SpeechDataset,
    'vertebral': VertebralDataset,
    'vowels': VowelsDataset,
    'wbc': WbcDataset,
    'wine': WineDataset,
    'seismic': SeismicDataset,
    'mulcross': MulcrossDataset,
    'ecoli': EcoliDataset,
    'cardio': CardioDataset,
    'fraud': FraudDataset,
    'backdoor': BackdoorDataset,
    'campaign': CampaignDataset,
}