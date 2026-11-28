"""Central dataset path configuration."""

FEMOT_ROOT = '/rydata/wangx_dataset/_dataset/FEMOT'
# RGB frames live in {seq}/{seq}_aps/; use annotations/{split}_rgb.json for YOLOX
FEMOT_RGB_DIR_SUFFIX = '_aps'
FEMOT_EVT_DIR_SUFFIX = '_dvs'
FEMOT_FRAME_FMT = '{frame_id:05d}.jpg'
FEMOT_TRAIN_ANN = 'train_rgb.json'
FEMOT_VAL_ANN = 'val_rgb.json'
FEMOT_TEST_ANN = 'test_rgb.json'
