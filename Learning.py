'''
Software Carpentry EN.540.635.01
Final Project: Antimicrobial peptide sequence prediction
May. 2024

Contributors:
Xiaoyuan Chen 

Please refer to the README.md file for more information.
'''

# Import packages
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300

def data_split(tot_pep, data):
    '''
    This function takes a dataframe "data" and splits it using the 
    original dataframe "tot_pep" as index reference.
    '''
    # Split the data to have 20% as final test and 80% as cross validation
    final_test = data.sample(frac=0.2, random_state=114514)
    ft_score = tot_pep.loc[final_test.index,:]['score'].tolist()
    cross_validation = data.drop(final_test.index)
    cv_score = tot_pep.drop(final_test.index)
    cross_validation_i = cross_validation.reset_index(drop=False)
    cv_score_i = cv_score.reset_index(drop=False)

    # Split the cross validation part randomly for five times. Each time
    # trianing data makes up 80% and test data makes up 20%.
    kf = KFold(n_splits=5, shuffle=True, random_state=114514)
    cv_train = []
    cv_test = []
    cv_score_train = []
    cv_score_test = []
    for i, (train_index, test_index) in enumerate(kf.split(cross_validation)):
        train = cross_validation_i.loc[train_index,:]
        train.columns.values[0] = 'index'
        train.set_index('index', inplace=True)
        test = cross_validation_i.loc[test_index,:]
        test.columns.values[0] = 'index'
        test.set_index('index', inplace=True)
        train_score = cv_score_i.loc[train_index,:]['score'].tolist()
        test_score = cv_score_i.loc[test_index,:]['score'].tolist()
        cv_train.append(train)
        cv_test.append(test)
        cv_score_train.append(train_score)
        cv_score_test.append(test_score)
    
    # Return all split data.
    return cv_train, cv_test, cv_score_train, cv_score_test, cross_validation, cv_score, final_test, ft_score

def kmer_process(tot_pep):
    '''
    This function transforms the peptide sequence information into 3mer 
    (3 continuous amino acid) with step size of 1 amino acid.
    The numeric representation of sequence becomes the count number of
    each possible 3mer.
    '''
    # Find all possible 3mer in all the sequences from the data
    all_seq = tot_pep.index.tolist()
    three_mer_list = []
    for seq in all_seq:
        for i in range(len(seq) - 3 + 1):
            three_mer = seq[i:i+3]
            if not three_mer in three_mer_list:
                three_mer_list.append(three_mer)

    # Create count matrix of 3mer for all sequences
    mat = np.zeros((len(all_seq), len(three_mer_list)))
    data = pd.DataFrame(mat, index=all_seq, columns=three_mer_list)
    for seq in all_seq:
        for i in range(len(seq) - 3 + 1):
            three_mer = seq[i:i+3]
            data.at[seq, three_mer] += 1   

    return data_split(tot_pep, data)

def index_process(tot_pep):
    '''
    This function transforms the peptide sequence information into numeric
    indices (each amino acid character symbol corresponds to one number).
    The numeric representation of sequence becomes a list of numbers.
    '''
    # Find all possible amino acid character symbols in all the sequences from the data
    all_seq = tot_pep.index.tolist()
    aa_list = []
    for seq in all_seq:
        seq_aa = list(seq)
        for aa in seq_aa:
            if not aa in aa_list:
                aa_list.append(aa)

    # Transform amino acid character symbol into number.
    aa_list_sort = sorted(aa_list)
    aa_to_num_dict = {char: index for index, char in enumerate(aa_list_sort)}
    data_new = []
    for seq in all_seq:
        seq_aa = list(seq)
        seq_num = []
        for aa in seq_aa:
            num = aa_to_num_dict[aa]
            seq_num.append(num)
        data_new.append(seq_num)

    # Equalize the length of all sequences and give empty place -1.
    data_new = pd.DataFrame(data_new).fillna(-1)
    data_new.index = tot_pep.index

    return data_split(tot_pep, data_new)

def pca(df):
    '''
    This function takes a numeric dataframe and calculates the pca out of it.
    '''
    scaling = StandardScaler()
    scaling.fit(df)
    Scaled_data = scaling.transform(df)

    principal = PCA(n_components = 100)
    principal.fit(Scaled_data)
    x = principal.transform(Scaled_data)
    x = pd.DataFrame(x)
    return x

def ML(cv_train, cv_test, cv_score_train, cv_score_test, cross_validation, cv_score, final_test, ft_score):
    '''
    This function uses split data to perform machines learning using different
    traditional models and return the r2 values as performances.
    '''
    # List of all models used
    models = {
        '0': LinearRegression(),
        '1': Ridge(random_state=114514),
        '2': Lasso(),
        '3': RandomForestRegressor(n_estimators=25, random_state=114514),
        '4': SVR(),
        '5': KNeighborsRegressor()
    }

    # For each model, learning is performed on 5 cross validation groups 
    # and the final test. The mean r2 of cross validation and final test
    # are stored.
    stats = []
    for key in models:
        model = models[key]
        r2_list = []
        print("In {} model:".format(model))
        print("Cross validation:")
        for i in range(5):
            train = cv_train[i]
            test = cv_test[i]
            train_score = cv_score_train[i]
            test_score = cv_score_test[i]
            if key == '0':
                train_pca = pca(train)
                test_pca = pca(test)
                model.fit(train_pca, train_score)
                pred = model.predict(test_pca)
                r2 = r2_score(test_score, pred)
                print("Fold {}: r2 = {}".format(i, r2))
                r2_list.append(r2)
            else:
                model.fit(train, train_score)
                pred = model.predict(test)
                r2 = r2_score(test_score, pred)
                print("Fold {}: r2 = {}".format(i, r2))
                r2_list.append(r2)

        r2_mean = np.mean(r2_list)
        print("Mean: r2 = {}".format(r2_mean))

        if key == '0':
            cv_pca = pca(cross_validation)
            ft_pca = pca(final_test)
            model = models[key]
            model.fit(cv_pca, cv_score['score'].tolist())
            pred = model.predict(ft_pca)
            r2_final = r2_score(ft_score, pred)
            print("Final test: r2 = {}".format(r2_final))
        else:
            model = models[key]
            model.fit(cross_validation, cv_score['score'].tolist())
            pred = model.predict(final_test)
            r2_final = r2_score(ft_score, pred)
            print("Final test: r2 = {}".format(r2_final))

        stats.append([r2_mean, r2_final])
    return stats

def create_mlp(num_kmers):
    '''
    This function creates a multilayer perceptron model.
    '''
    inputs = tf.keras.layers.Input(shape=(num_kmers,))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mean_squared_error', metrics=[tf.keras.metrics.R2Score()])
    return model

def MLP(cv_train, cv_test, cv_score_train, cv_score_test, cross_validation, cv_score, final_test, ft_score):
    '''
    This function uses split data to perform neural network using multilayer
    perceptron model and return the r2 values as performances.
    '''
    kmer_r2_list = []
    for i in range(5):
        train = cv_train[i]
        test = cv_test[i]
        train_score = cv_score_train[i]
        test_score = cv_score_test[i]

        model_kmer = create_mlp(9752)
        model_kmer.fit(train, pd.DataFrame(train_score), epochs=7, batch_size=32, validation_data=(test, pd.DataFrame(test_score)))
        pred = model_kmer.predict(test)
        r2 = r2_score(test_score, pred)
        print("Fold {}: r2 = {}".format(i, r2))
        kmer_r2_list.append(r2)

    kmer_r2_mean = np.mean(kmer_r2_list)
    print("Mean: r2 = {}".format(kmer_r2_mean))

    model_kmer = create_mlp(9752)
    model_kmer.fit(cross_validation, pd.DataFrame(cv_score['score'].tolist()), epochs=7, batch_size=16, validation_data=(final_test, pd.DataFrame(ft_score)))
    pred = model_kmer.predict(final_test)
    kmer_r2_final = r2_score(ft_score, pred)
    print("Final test: r2 = {}".format(kmer_r2_final))

    return kmer_r2_mean, kmer_r2_final

def create_model():
    '''
    This function creates a convolutional recurrent model.
    '''
    inputs = tf.keras.layers.Input(shape=(50, ))
    x = tf.keras.layers.Reshape((50, 1))(inputs)
    x = tf.keras.layers.Conv1D(1024, 6, 1, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mean_squared_error', metrics=[tf.keras.metrics.R2Score()])
    return model

def CRNN(cv_train_new, cv_test_new, cv_score_train_new, cv_score_test_new, cross_validation_new, cv_score_new, final_test_new, ft_score_new):
    '''
    This function uses split data to perform neural network using convolutional
    recurrent model and return the r2 values as performances.
    '''
    index_r2_list = []
    for i in range(5):
        train = cv_train_new[i]
        test = cv_test_new[i]
        train_score = cv_score_train_new[i]
        test_score = cv_score_test_new[i]

        model_index = create_model()
        model_index.fit(train, pd.DataFrame(train_score), epochs=15, batch_size=32, validation_data=(test, pd.DataFrame(test_score)))
        pred = model_index.predict(test)
        r2 = r2_score(test_score, pred)
        print("Fold {}: r2 = {}".format(i, r2))
        index_r2_list.append(r2)

    index_r2_mean = np.mean(index_r2_list)
    print("Mean: r2 = {}".format(index_r2_mean))

    model_index = create_model()
    model_index.fit(cross_validation_new, pd.DataFrame(cv_score_new['score'].tolist()), epochs=15, batch_size=32, validation_data=(final_test_new, pd.DataFrame(ft_score_new)))
    pred = model_index.predict(final_test_new)
    index_r2_final = r2_score(ft_score_new, pred)
    print("Final test: r2 = {}".format(index_r2_final))

    return index_r2_mean, index_r2_final

def Compare(stats):
    '''
    This functions takes the r2 statistics (cross validation + final test)
    of machine learning results and show the performance comparison in barplot. 
    '''
    model_list = ['Linear', 'Ridge', 'Lasso', 'RF', 'SVR', 'KNN', 'MLP', 'CRNN']
    group_labels = []
    categories = []
    for i in range(8):
        group_label = model_list[i]
        group_labels.extend([group_label] * 2)
        categories.append('CV_mean')
        categories.append('FT')
    flat_stats = [value for group in stats for value in group]
    stat_data = pd.DataFrame({
        'Model': group_labels,
        'r2': flat_stats,
        'Category': categories
    })

    sns.barplot(x='Model', y='r2', hue='Category', data=stat_data, errorbar=None)
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('R2', fontsize=14)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, title_fontsize=12)
    plt.tight_layout()
    plt.savefig('Performance_Comparison.png', bbox_inches='tight')
    plt.clf()

def Compare_true(cross_validation, cv_score, final_test, ft_score, model, title, ANN=False):
    '''
    This function takes the whole validation data and uses it to fit the chosen
    model. The model then predicts the values from the final test data, which 
    are then compared to the true final test scores.
    '''
    model_1 = model
    if not ANN:
        model_1.fit(cross_validation, cv_score['score'].tolist())
    else:
        model_1.fit(cross_validation, pd.DataFrame(cv_score['score'].tolist()), epochs=7, batch_size=16)
    pred_1 = model_1.predict(final_test)

    meta_1 = LinearRegression()
    meta_1.fit(pred_1.reshape(-1, 1), np.array(ft_score).reshape(-1, 1))
    line = meta_1.predict(pred_1.reshape(-1, 1))
    fig, axs = plt.subplots(1,1)
    axs.scatter(pred_1, ft_score, marker="+", color="royalblue", alpha=0.5)
    axs.plot(pred_1, line, color="black")
    axs.set_xlabel("Predicted")
    axs.set_ylabel("True")
    plt.title(title)
    plt.savefig(title + ".png")
    plt.clf()


if __name__ == '__main__':
    '''
    Data Preprocessing
    Information of active peptides and inactive peptides is seperated loaded and merged together.
    Peptides are filtered based on sequence length distribution.
    '''
    # Data loading
    act_pep = pd.read_csv("AMP0_data.csv")
    act_pep = act_pep[act_pep['Target species'] == 'Escherichia coli']
    act_pep = act_pep.drop_duplicates(subset=['Sequence'])

    ina_file = "AMPlify_non_AMP_imbalanced.fa"
    ina_pep = []
    with open(ina_file, "r") as f:
        for line in f:
            if not line.startswith(">"):
                ina_pep.append(line.split("\n")[0])

    # Distribution analysis
    act_len = []
    for index, row in act_pep.iterrows():
        seq = row['Sequence']
        act_len.append(len(seq))

    ina_len = []
    for pep in ina_pep:
        ina_len.append(len(pep))

    sns.histplot(ina_len, color="blue")
    sns.histplot(act_len, color="red")
    fig = plt.gcf()
    fig.savefig('Peptide_Length_Distribution.png')
    plt.clf()

    # Filter based on distribution
    mask = act_pep['Sequence'].str.len() <= 50
    act_pep_fil = act_pep[mask]

    ina_pep_fil = [s for s in ina_pep if len(s) <= 50]

    act_len_fil = []
    for index, row in act_pep_fil.iterrows():
        seq = row['Sequence']
        act_len_fil.append(len(seq))

    ina_len_fil = []
    for pep in ina_pep_fil:
        ina_len_fil.append(len(pep))

    sns.histplot(ina_len_fil, color="blue")
    sns.histplot(act_len_fil, color="red")
    fig = plt.gcf()
    fig.savefig('Filtered_Peptide_Length_Distribution.png')
    plt.clf()

    # Data merge
    tot_pep = act_pep_fil[['Sequence', 'MIC (?g/mL )']]
    tot_pep = tot_pep.copy()
    tot_pep.loc[:, 'score'] = tot_pep['MIC (?g/mL )'] / (tot_pep['MIC (?g/mL )'] + 128)

    ina_data = {
        'Sequence': ina_pep_fil,
        'MIC (?g/mL )': ["unknown"] * len(ina_pep_fil),
        'score': [1] * len(ina_pep_fil)
    }
    ina_df = pd.DataFrame(ina_data)
    tot_pep = pd.concat([tot_pep, ina_df])
    tot_pep = tot_pep.drop_duplicates(subset=['Sequence'])
    tot_pep.set_index('Sequence', inplace=True)

    # Machine learning (traditional model + neural network)
    cv_train, cv_test, cv_score_train, cv_score_test, cross_validation, cv_score, final_test, ft_score = kmer_process(tot_pep)
    cv_train_new, cv_test_new, cv_score_train_new, cv_score_test_new, cross_validation_new, cv_score_new, final_test_new, ft_score_new = index_process(tot_pep)
    stats = ML(cv_train, cv_test, cv_score_train, cv_score_test, cross_validation, cv_score, final_test, ft_score)
    kmer_r2_mean, kmer_r2_final = MLP(cv_train, cv_test, cv_score_train, cv_score_test, cross_validation, cv_score, final_test, ft_score)
    index_r2_mean, index_r2_final = CRNN(cv_train_new, cv_test_new, cv_score_train_new, cv_score_test_new, cross_validation_new, cv_score_new, final_test_new, ft_score_new)

    # Performance comparison
    stats.append([kmer_r2_mean, kmer_r2_final])
    stats.append([index_r2_mean, index_r2_final])
    Compare(stats)

    # Best performance show
    Compare_true(cross_validation, cv_score, final_test, ft_score, SVR(), "SVR")
    Compare_true(cross_validation, cv_score, final_test, ft_score, create_mlp(9752), "MLP")
    
