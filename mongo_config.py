config = {
    # +
    # This is where you specify your dataset:
    # whether it's CVS or MongoDB, the connect parameters, etc.
    #
    # To experiment with this, replace the CSV file with any dataset that you
    # wish to analyze.  Or point this to a MongoDB collection.
    #
    # This particular dataset contains records indicating success or failure of an Air Force 
    # Reserve recruitment effort on a per-recruit basis.
    #
    'data': {
        'type': 'MongoDB',
        'collection': 'FY11Accessions',
        'dbname': 'pivotDb',
        'query': {},
        'host': 'localhost',
        'port': 27017,
        'username': None,
        'password': None,
        'noid': True,
    },

    # This object describes the features and labels of the collection above.
    #
    'collection': {
        # This is the human-readable (success, failure) message vector.
        #
        'label_outputs': ['Recruitment Failure', 'Recruitment Success'],

        # Specify the categorical features from the dataset that
        # we hypothesize may affect successful recruitment.
        # 
        # We wish to see if successful recruitment is predictable based on gender, race, or state;
        # you should feel free to change these features in order to experiment
        # with other hypotheses, or just remove ones you hypothesize are not
        # relevant, etc.  This kind of experimentation can be very enlightening.
        #
        # To experiment with your own dataset, change this so it contains whatever features you
        # think might influence the label in your own dataset.
        #
        # However, NOTE WELL that some (many!) datasets contain "linear duplicates" 
        # where a (feature) column is semantically related to the label column.  
        # If that's the case, then do NOT include the semantically overlapping 
        # feature column here.  You'll get 100% success in your model, but
        # the result will be meaningless.
        #
        'cat_features': ['GENDER', 'RACE', 'STATE'],

        # If your dataset contains continuous variables (like age, for example),
        # add them in here.
        #
        # For the Air Force dataset, there aren't any continuous variables.  However,
        # for demonstration purposes, I'll specify that the zip code is actually a
        # numeric value.  The model should learn to ignore the variable (unless, for
        # some unforeseen reason, the magnitude of a zip code affects recruitment
        # success.  Color me skeptical.)
        #
        'cont_features': ['ZIP'],

        # The CURSTATUS_CD field contains values indicating the outcome of an Air Force
        # recruitment effort.
        #
        # To experiment with this, replace this with the label from your own dataset.
        #
        'label_field': 'CURSTATUS_CD',

        # Declare which values of the CURSTATUS_CD field are considered positive outcomes
        # (that is, successful recruitment)
        #
        # ACE = accession gain, ACG = accession gain, as per prior information request
        #
        # To experiment with this, change this so it contains the positive values from
        # the label_field in your dataset.
        #
        'true_values': ['ACE', 'ACG'],
    },

    # These model parameters are the ones that may change fairly frequently
    #
    'model': {
        'num_epochs': 35,
        'batch_size': 4,
        'learning_rate': 0.001
    }
}
