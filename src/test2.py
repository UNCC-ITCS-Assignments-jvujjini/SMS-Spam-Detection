def features_from_messages(messages, label, feature_extractor, **kwargs):
    '''
    Make a (features, label) tuple for each message in a list of a certain,
    label of e-mails ('spam', 'ham') and return a list of these tuples.

    Note every e-mail in 'messages' should have the same label.
    '''
    features_labels = []
    for msg in messages:
        features = feature_extractor(msg, **kwargs)
        features_labels.append((features, label))
    return features_labels

def word_indicator(msg, **kwargs):
    '''
    Create a dictionary of entries {word: True} for every unique
    word in a message.

    Note **kwargs are options to the word-set creator,
    get_msg_words().
    '''
    features = defaultdict(list)
    #msg_words = get_msg_words(msg, **kwargs)
    for  w in msg:
            features[w] = True
    return features

def make_train_test_sets(feature_extractor, **kwargs):
    '''
    Make (feature, label) lists for each of the training 
    and testing lists.
    '''
    train_spam = features_from_messages(TrainSMSSpam, 'spam', 
                                        feature_extractor, **kwargs)
    train_ham = features_from_messages(TrainSMSHam, 'ham', 
                                       feature_extractor, **kwargs)
    
    train_set = train_spam + train_ham

    test_spam = features_from_messages(TestSMSSpam, 'spam',
                                       feature_extractor, **kwargs)

    test_ham = features_from_messages(TestSMSHam, 'ham',
                                      feature_extractor, **kwargs)
    
    return train_set, test_spam, test_ham

#def trainNaiveBayes(testSample):
    
def check_classifier(feature_extractor, **kwargs):
    '''
    Train the classifier on the training spam and ham, then check its accuracy
    on the test data, and show the classifier's most informative features.
    '''
    
    # Make training and testing sets of (features, label) data
    train_set, test_spam, test_ham = make_train_test_sets(feature_extractor, **kwargs)
    
    # Train the classifier on the training set
    classifier = NaiveBayesClassifier.train(train_set)
    
    # How accurate is the classifier on the test sets?
    print ('Test Spam accuracy: {0:.2f}%'
       .format(100 * nltk.classify.accuracy(classifier, test_spam)))
    print ('Test Ham accuracy: {0:.2f}%'
       .format(100 * nltk.classify.accuracy(classifier, test_ham)))

    # Show the top 20 informative features
    print classifier.show_most_informative_features(20)

check_classifier(word_indicator)