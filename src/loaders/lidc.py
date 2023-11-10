from datasets.LIDC_loader import LIDC_loader
from torch.utils.data import DataLoader, Subset
from modules.transforms import DiffusionTransform, DataAugmentationTransform
import albumentations as A
import glob



def get_lidc(config, logger=None, verbose=False):

    if logger: print = logger.info

    INPUT_SIZE = config["dataset"]["input_size"]
    DT = DiffusionTransform((INPUT_SIZE, INPUT_SIZE))
    AUGT = DataAugmentationTransform((INPUT_SIZE, INPUT_SIZE))
    
    img_dir = "Images"
    msk_dir = "Masks"
    img_path_list = glob.glob(f"{config['dataset']['data_dir']}/{img_dir}/*.png")
    
    pixel_level_transform = AUGT.get_pixel_level_transform(config["augmentation"], img_path_list=img_path_list)
    spacial_level_transform = AUGT.get_spacial_level_transform(config["augmentation"])
    tr_aug_transform = A.Compose([
        A.Compose(pixel_level_transform, p=config["augmentation"]["levels"]["pixel"]["p"]), 
        A.Compose(spacial_level_transform, p=config["augmentation"]["levels"]["spacial"]["p"])
    ], p=config["augmentation"]["p"])

    # ----------------- dataset --------------------
    if config["dataset"]["class_name"] == "LIDC_loader":

        dataset = LIDC_loader(
            data_dir=config["dataset"]["data_dir"],
            one_hot=False,
            # aug=AUGT.get_aug_policy_3(),
            # transform=AUGT.get_spatial_transform(),
            img_transform=DT.get_forward_transform_img(),
            msk_transform=DT.get_forward_transform_msk(),
            logger=logger
        )
        
        length = len(dataset)
        
        tr_range = range(0, int(length * 0.8))
        vl_range = range(int(length * 0.8), int(length * 0.9))
        te_range = range(int(length * 0.9), length)

        
        tr_dataset = Subset(dataset, tr_range)
        vl_dataset = Subset(dataset, vl_range)
        te_dataset = Subset(dataset, te_range)
        # We consider 7200 samples for training, 1800 samples for validation and 1015 samples for testing
        # !cat ~/deeplearning/skin/Prepare_HAM10000.py
    else:
        message = "In the config file, `dataset>class_name` should be in: ['HAM10000Dataset', 'HAM10000DatasetFast']"
        if logger: 
            logger.exception(message)
        else:
            raise ValueError(message)

    if verbose:
        print("LIDC:")
        print(f"├──> Length of trainig_dataset:\t   {len(tr_dataset)}")
        print(f"├──> Length of validation_dataset: {len(vl_dataset)}")
        print(f"└──> Length of test_dataset:\t   {len(te_dataset)}")

    # prepare train dataloader
    tr_dataloader = DataLoader(tr_dataset, **config["data_loader"]["train"])

    # prepare validation dataloader
    vl_dataloader = DataLoader(vl_dataset, **config["data_loader"]["validation"])

    # prepare test dataloader
    te_dataloader = DataLoader(te_dataset, **config["data_loader"]["test"])

    return {
        "tr": {"dataset": tr_dataset, "loader": tr_dataloader},
        "vl": {"dataset": vl_dataset, "loader": vl_dataloader},
        "te": {"dataset": te_dataset, "loader": te_dataloader},
    }


# # test and visualize the input data
# from utils.helper_funcs import show_sbs
# for sample in tr_dataloader:
#     img = sample['image']
#     msk = sample['mask']
#     print("Training")
#     print(img.shape, msk.shape)
#     show_sbs(img[0], msk[0])
#     break

# for sample in vl_dataloader:
#     img = sample['image']
#     msk = sample['mask']
#     print("Validation")
#     show_sbs(img[0], msk[0])
#     break

# for sample in te_dataloader:
#     img = sample['image']
#     msk = sample['mask']
#     print("Test")
#     show_sbs(img[0], msk[0])
#     break
