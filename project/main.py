import src.data_processing.processing_pipeline as dp
import src.models.model_pipeline as mp

def main():

    # data processing 
    dp.main_pipeline()

    # model training and evaluation
    #mp.main_pipeline()
    
if __name__ == '__main__':
    main()
