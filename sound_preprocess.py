import os 
import librosa
import math
import json


# dataset_path --> is the path of the  dataset
# json_path    --> is the path where we want to store our mfcc data. 
# num_segments --> we are dividing/breaking our track/sound into multiple segments so that it will become computationally 
# less expensive 
# In this way we will have more input data
# SAMPLES_PER_TRACK --> shows the overall sample available in our audio track.
# SAMPLE_PER_TRACK is recorded by * duration with overall samples in a track.
# DURATION --> duration of our sound, overhere our sound is of 30 secs length. 

DATASET_PATH = "genres"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
DURATION = 30  # measured in seconds.
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, n_mfcc = 13, n_fft = 2048, hop_length= 512, num_segments = 5):
   
    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    
    num_samples_per_segment =  int(SAMPLES_PER_TRACK / num_segments)
    # this shows how many samples/amp points/freqcs we have in each segment 
    # num_samples_per_segment is recorded by dividing overall samples per track with total no of segments per track.
    
    expected_no_of_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length) 
    # the result of this can be floating value so we have to round it up to the highest no e.g. 1.2 --> 2
    # for this we use ceil()
    
    
    # mapping --> we need to map different genres on to a number, so that each genre represents a number. 
    # mfcc --> we r gonna have mfcc vector for each segment. so for 5 segments we will have 5 mfcc vectors, each mfcc 
    # represents a window, in which we have no of samples.
    # so for each image we will have 5 samples so it means 5 mfcc vectors. 
    # label --> labels are the output of each mfcc vector. 
    
    # 1) Loop through all the genres
    for i, (dirpath, dirname, filenames) in enumerate(os.walk(dataset_path)):
    # dirpath is the current path/folder we r in.
    # dirname are all the name of the subfolders. 
    # filenames are all the files that we have in dirpath.
        
        # Ensure that we r not at the root level
        if dirpath is not dataset_path:
            # save the semantic labels
            # sematic labels --> mapping contains semantic labels. e.g on 0 we have classical, on 1 we have blues etc
            dirpath_components = dirpath.split("\\")  # "genre/blues" => ["genre", "blues"]
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            
            print("\n Processing {}".format(semantic_label))
            
            # Process files for specific genre. 
            for f in filenames:
                # f just give us the name of the file it doesnt give us the full path.
                # we need full path for loading the file so for loading full path we do
                
                # laoding audio file
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr = SAMPLE_RATE)
                # we can't anaylyze mfcc's at this level blc we have broken our song into small chunks/segments
                # so we have to analyze mfcc's at segment level. so for this, we have to divide our signal/sound into bunch
                # of segments.
                
                # divide signals into segments,process segments, extract mfcc and at last store mfcc vectors.
                for s in range(num_segments):
                    # for each segment in a signal/sound we need start sample and finish sample value.
                    start_sample = num_samples_per_segment * s  # if s= 0 -> then start_sample = 0 
                    finish_sample = start_sample + num_samples_per_segment  # if s=0 -> num_segments_per_sample
                    
                    mfcc = librosa.feature.mfcc(signal[start_sample: finish_sample],
                                                sr =sr,
                                                n_fft = n_fft,
                                                n_mfcc = n_mfcc,
                                                hop_length = hop_length
                                                )
                    # so we dont want to analyze the whole signal but we want to analyze the slice of that signal. 
                    # so it is gonna be blw start sample and finish sample.
                    
                    mfcc = mfcc.T   # making mfcc vector suitable to our DL model.
                    # when we do mfcc we may have more vector in one segment that the other. i.e. inequility in shape
                    # so we need to make sure that every mfcc vector for each segment must be of the same shape.
                    # remember we need to make sure that we have same no of mfcc vectors for each segment. 
                    # all the segment length and shape should be same. For this we r gonna calcualte
                    # expected_no_of_mfcc_vectors_per_segment.
                    
                    # store mfcc vector for each segment, if it has the expected length. 
                    if len(mfcc) == expected_no_of_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist()) # we can not append mfcc directly blc its a numpy array so we have
                        # to first convert it into list.
                        
                        data["labels"].append(i-1)
                        # i mean in each iteration we are in different genre folder. why r we doing i -1 that's blc in first 
                        # iteration we were not inside the dir containing all the genres folder. 
                        
                        print("{}, segment : {}".format(file_path, s))
                
    # final step: saving everything as a json file            
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent = 4)  # indent mean spaces while writing. fp mean file_path  
        
        
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments = 10)
                