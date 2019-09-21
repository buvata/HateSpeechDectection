**Step by step to create classifier**

-   First step in project is get important keyword
-   Then make a dict synonym with important keyword
-   Make augmentation data with file handle_augmentation

-   ## Train model with ML:
    - Create data save by joblib with feature extract module.
    - Train model with number of ml model with setting for gridsearchcv
-   ## Train model with DL:
    - we have 3 model:
        -   cnn word + cnn char (char whole sequence)
        -   lstm word + cnn char (char whole sequence)
        -   lstm word + cnn/lstm char (char base word)    
    - train model with each model.
       
    
     