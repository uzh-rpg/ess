import glob
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy


def load_files_in_directory(directory, t_interval=50):
    # Load files: these include these have the following form
    #
    # idx.npy :
    #   t0_ns idx0
    #   t1_ns idx1
    #   ...
    #   tj_ns idxj
    #   ...
    #   tN_ns idxN
    #
    # This file contains a mapping from j -> tj_ns idxj,
    # where j+1 is the idx of the img with timestamp tj_ns (in nanoseconds)
    # and idxj is the idx of the last event before the img (in events.dat.t and events.dat.xyp)
    if t_interval == 10:
        img_timestamp_event_idx = np.load(os.path.join(directory, "index/index_10ms.npy"))
    elif t_interval == 50:
        img_timestamp_event_idx = np.load(os.path.join(directory, "index/index_50ms.npy"))
    elif t_interval == 250:
        img_timestamp_event_idx = np.load(os.path.join(directory, "index/index_250ms.npy"))
    else:
        img_timestamp_event_idx = np.load(os.path.join(directory, "index/index_50ms.npy"))

    # events.dat.t :
    #   t0_ns
    #   t1_ns
    #   ...
    #   tM_ns
    #
    #  events.dat.xyp :
    #    x0 y0 p0
    #    ...
    #    xM yM pM
    events_t_file = os.path.join(directory, "events.dat.t")
    events_xyp_file = os.path.join(directory, "events.dat.xyp")

    t_events, xyp_events = load_events(events_t_file, events_xyp_file)

    # Since the imgs are in a video format, they cannot be loaded directly, however, the segmentation masks from the
    # original dataset (EV-SegNet) have been copied into this folder. First unzip the segmentation masks with
    #
    #    unzip segmentation_masks.zip
    #
    segmentation_mask_files = sorted(glob.glob(os.path.join(directory, "segmentation_masks", "*.png")))

    return img_timestamp_event_idx, t_events, xyp_events, segmentation_mask_files


def load_events(t_file, xyp_file):
    # events.dat.t saves the timestamps of the indiviual events (in nanoseconds -> int64)
    # events.dat.xyp saves the x, y and polarity of events in uint8 to save storage. The polarity is 0 or 1.
    # We first need to compute the number of events in the memmap since it does not do it for us. We can do
    # this by computing the file size of the timestamps and dividing by 8 (since timestamps take 8 bytes)

    num_events = int(os.path.getsize(t_file) / 8)
    t_events = np.memmap(t_file, dtype="int64", mode="r", shape=(num_events, 1))
    xyp_events = np.memmap(xyp_file, dtype="int16", mode="r", shape=(num_events, 3))

    return t_events, xyp_events


def extract_events_from_memmap(t_events, xyp_events, img_idx, img_timestamp_event_idx, fixed_duration=False,
                               nr_events=32000):
    # timestep, event_idx = img_timestamp_event_idx[img_idx]
    # _, event_idx_before = img_timestamp_event_idx[img_idx - 1]
    if fixed_duration:
        timestep, event_idx, event_idx_before = img_timestamp_event_idx[img_idx]
        event_idx_before = max([event_idx_before, 0])
    else:
        timestep, event_idx, _ = img_timestamp_event_idx[img_idx]
        event_idx_before = max([event_idx - nr_events, 0])
    events_between_imgs = np.concatenate([
        np.array(t_events[event_idx_before:event_idx], dtype="int64"),
        np.array(xyp_events[event_idx_before:event_idx], dtype="int64")
    ], -1)
    events_between_imgs = events_between_imgs[:, [1, 2, 0, 3]]  # events have format xytp, and p is in [0,1]

    return events_between_imgs


def generate_event_img(shape, events):
    H, W = shape
    # generate event img
    event_img_pos = np.zeros((H * W,), dtype="float32")
    event_img_neg = np.zeros((H * W,), dtype="float32")

    x, y, t, p = events.T

    np.add.at(event_img_pos, x[p == 1] + W * y[p == 1], p[p == 1])
    np.add.at(event_img_neg, x[p == 0] + W * y[p == 0], p[p == 0] + 1)

    event_img_pos = event_img_pos.reshape((H, W))
    event_img_neg = event_img_neg.reshape((H, W))

    return event_img_neg, event_img_pos


def generate_colored_label_img(shape, label_mask):
    H, W = shape

    colors = [[0, 0, 255], [255, 0, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]
    mask = segmentation_mask.reshape((-1, 3))[:, 0]

    img = np.zeros((H * W, 3), dtype="uint8")

    for i in np.unique(segmentation_mask):
        c = colors[int(i)]
        img[mask == i, 0] = c[0]
        img[mask == i, 1] = c[1]
        img[mask == i, 2] = c[2]

    img = img.reshape((H, W, 3))

    return img


def generate_rendered_events_on_img(img, event_map_neg, event_map_pos):
    orig_shape = img.shape

    img = img.copy()

    img = img.reshape((-1, 3))
    pos_mask = event_map_pos.reshape((-1,)) > 0
    neg_mask = event_map_neg.reshape((-1,)) > 0

    img[neg_mask, 0] = 255
    img[pos_mask, 2] = 255
    img[neg_mask | pos_mask, 1] = 0

    img = img.reshape(orig_shape)

    return img


if __name__ == "__main__":

    directories = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "dir*")))
    assert len(directories) > 0
    # example with one directory
    directory = directories[1]
    print(directories)
    print("Using directory: %s" % directory)

    # load all files that are in the directory (these are real events)
    img_timestamp_event_idx, t_events, xyp_events, segmentation_mask_files = \
        load_files_in_directory(directory)

    # load all files that are in video_upsampled_events. These are simulated data.
    sim_directory = os.path.join(directory, "video_upsampled_events")
    load_sim = os.path.exists(sim_directory)
    img_timestamp_event_idx_sim, t_events_sim, xyp_events_sim = None, None, None
    if load_sim:
        print("Loading sim data")
        img_timestamp_event_idx_sim, t_events_sim, xyp_events_sim, _ = \
            load_files_in_directory(sim_directory)

    num_plots = 3 if load_sim else 2
    fig, ax = plt.subplots(ncols=num_plots)
    img_handles = []
    assert len(segmentation_mask_files) > 0
    for segmentation_mask_file in segmentation_mask_files[-100:]:
        # take an example mask and extract the corresponding idx
        print("Using segmentation mask: %s" % segmentation_mask_file)
        segmentation_mask = cv2.imread(segmentation_mask_file)

        img_idx = int(os.path.basename(segmentation_mask_file).split("_")[-1].split(".")[0]) - 1
        print("Loading img with idx %s" % img_idx)

        # load corresponding img
        # first decompress video by running
        #
        #    mkdir imgs
        #    ffmpeg -i video.mp4  imgs/img_%08d.png
        #
        img_file = segmentation_mask_file.replace("segmentation_masks", "imgs").replace("/segmentation_", "/img_")
        # img_file = '/'.join(img_file.split('/')[:-1]) + '/' + '{:0>10}'.format(img_file.split('_')[-1][:-4]) + '.png'
        img = cv2.imread(img_file)

        # crop img since this was done in EV-SegNet
        img = img[:200]

        # find events between this idx and the last
        events_between_imgs = \
            extract_events_from_memmap(t_events, xyp_events, img_idx, img_timestamp_event_idx)
        print("Found %s events" % (len(events_between_imgs)))

        # remove all events with y > 200 since these were cropped from the dataset
        events_between_imgs = events_between_imgs[events_between_imgs[:, 1] < 200]

        if load_sim:
            # find sim events between this idx and the last
            events_between_imgs_sim = \
                extract_events_from_memmap(t_events_sim, xyp_events_sim, img_idx, img_timestamp_event_idx_sim)
            print("Found %s simulated events" % (len(events_between_imgs_sim)))

            # remove all events with y > 200 since these were cropped from the dataset
            events_between_imgs_sim = events_between_imgs_sim[events_between_imgs_sim[:, 1] < 200]

        event_img_neg, event_img_pos = generate_event_img((200, 346), events_between_imgs)
        event_img_neg_sim, event_img_pos_sim = generate_event_img((200, 346), events_between_imgs_sim)

        # generate view of labels
        colored_label_img = generate_colored_label_img((200, 346), segmentation_mask)

        # draw events on img
        rendered_events_on_img = generate_rendered_events_on_img(copy.deepcopy(img), event_img_neg, event_img_pos)

        if load_sim:
            # draw events on img
            rendered_events_on_img_sim = generate_rendered_events_on_img(copy.deepcopy(img), event_img_neg_sim,
                                                                         event_img_pos_sim)

        print("Error: ",
              np.abs((rendered_events_on_img_sim).astype("float32") - (rendered_events_on_img).astype("float32")).sum())

        if len(img_handles) == 0:
            img_handles += [ax[0].imshow(colored_label_img)]
            img_handles += [ax[1].imshow(rendered_events_on_img)]
            if load_sim:
                img_handles += [ax[2].imshow(rendered_events_on_img_sim)]
            plt.show(block=False)
        else:
            img_handles[0].set_data(colored_label_img)
            img_handles[1].set_data(rendered_events_on_img)
            if load_sim:
                img_handles[2].set_data(rendered_events_on_img_sim)
            fig.canvas.draw()
            plt.pause(0.002)


