import pandas as pd

def manual_label_smoothing(y):
    y1, y2= y[0], y[1]

    ### ALL CERTAIN ###
    # both have driveways
    if y1 ==1.0 and y2 == 1.0:
        return 1.0
    # both don't have driveways
    if y1 == 0.0 and y2 == 0.0:
        return 0.0
    # aerial has driveway, streetview doesn't have driveway
    if y1 ==1.0 and y2 == 0.0:
        return 0.9
    # aerial doesn't have driveway, streetview has driveway
    if y1 == 0.0 and y2 == 1.0:
        return 0.9

    ### ALL UNCERTAIN ###
    if y1 == 2.0 and y2 == 2.0:
        return 0.5

    ### ONE UNCERTAIN ###

    if y1 == 2.0 and y2 != 2.0:
        if y2 == 1.0:
            return 0.9
        if y2 == 0.0:
            return 0.1

    if y1 != 2.0 and y2 == 2.0:
        if y1 == 1.0:
            return 0.9
        if y1 == 0.0:
            return 0.1

if __name__ == "__main__":
    # this is the google drive file downloaded
    sample = pd.read_csv('../training/new_sample_110319.csv', index_col= 0)
    # only keep rows with labels
    sample = sample[~sample.AERIAL_Driveway.isna()]
    sample = sample[~sample.GSV_Driveway.isna()]
    # get labels
    sample['final_label'] = sample[['AERIAL_Driveway','GSV_Driveway']].apply(manual_label_smoothing, axis = 1)
    # get filenames for the generator
    sample['aerial_filename'] = sample['ADDR_NUM'] + '_' + sample['FULL_STR'].str.replace(' ', '_') + '_aerial.png'
    sample['gsv_filename'] = sample['ADDR_NUM'] + '_' + sample['FULL_STR'].str.replace(' ', '_') + '.jpg'
