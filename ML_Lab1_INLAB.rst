.. code:: ipython3

    import numpy as np
    
    # Creating Arrays
    arr1 = np.array([10, 20, 30, 40, 50])
    print("1D Array:", arr1)
    
    arr2 = np.array([[1, 2, 3],
                     [4, 5, 6]])
    print("2D Array:\n", arr2)
    
    # Special Arrays
    print(np.zeros((2,3)))
    print(np.ones((3,3)))
    print(np.eye(4))
    print(np.arange(1,11))
    print(np.arange(2,21,2))
    print(np.linspace(0,10,5))
    
    # Properties
    print(arr2.shape)
    print(arr2.size)
    print(arr2.ndim)
    print(arr2.dtype)
    
    # Mathematical Operations
    print(arr1+5)
    print(arr1-5)
    print(arr1*2)
    print(arr1/2)
    print(arr1**2)
    print(np.sqrt(arr1))
    
    # Statistics
    print(np.sum(arr1))
    print(np.mean(arr1))
    print(np.median(arr1))
    print(np.max(arr1))
    print(np.min(arr1))
    print(np.std(arr1))
    print(np.var(arr1))


.. parsed-literal::

    1D Array: [10 20 30 40 50]
    2D Array:
     [[1 2 3]
     [4 5 6]]
    [[0. 0. 0.]
     [0. 0. 0.]]
    [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
    [ 1  2  3  4  5  6  7  8  9 10]
    [ 2  4  6  8 10 12 14 16 18 20]
    [ 0.   2.5  5.   7.5 10. ]
    (2, 3)
    6
    2
    int32
    [15 25 35 45 55]
    [ 5 15 25 35 45]
    [ 20  40  60  80 100]
    [ 5. 10. 15. 20. 25.]
    [ 100  400  900 1600 2500]
    [3.16227766 4.47213595 5.47722558 6.32455532 7.07106781]
    150
    30.0
    30.0
    50
    10
    14.142135623730951
    200.0
    

.. code:: ipython3

    import pandas as pd
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    
    df['Species'] = iris.target
    
    df['Species'] = df['Species'].map({
        0:'Setosa',
        1:'Versicolor',
        2:'Virginica'
    })
    
    print(df.head())
    print(df.tail())
    print(df.shape)
    print(df.columns)
    print(df.dtypes)
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    print(df.duplicated().sum())
    print(df.sample(5))
    print(df.nunique())
    print(df['Species'].value_counts())


.. parsed-literal::

       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \
    0                5.1               3.5                1.4               0.2   
    1                4.9               3.0                1.4               0.2   
    2                4.7               3.2                1.3               0.2   
    3                4.6               3.1                1.5               0.2   
    4                5.0               3.6                1.4               0.2   
    
      Species  
    0  Setosa  
    1  Setosa  
    2  Setosa  
    3  Setosa  
    4  Setosa  
         sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \
    145                6.7               3.0                5.2               2.3   
    146                6.3               2.5                5.0               1.9   
    147                6.5               3.0                5.2               2.0   
    148                6.2               3.4                5.4               2.3   
    149                5.9               3.0                5.1               1.8   
    
           Species  
    145  Virginica  
    146  Virginica  
    147  Virginica  
    148  Virginica  
    149  Virginica  
    (150, 5)
    Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
           'petal width (cm)', 'Species'],
          dtype='object')
    sepal length (cm)    float64
    sepal width (cm)     float64
    petal length (cm)    float64
    petal width (cm)     float64
    Species               object
    dtype: object
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   sepal length (cm)  150 non-null    float64
     1   sepal width (cm)   150 non-null    float64
     2   petal length (cm)  150 non-null    float64
     3   petal width (cm)   150 non-null    float64
     4   Species            150 non-null    object 
    dtypes: float64(4), object(1)
    memory usage: 6.0+ KB
    None
           sepal length (cm)  sepal width (cm)  petal length (cm)  \
    count         150.000000        150.000000         150.000000   
    mean            5.843333          3.057333           3.758000   
    std             0.828066          0.435866           1.765298   
    min             4.300000          2.000000           1.000000   
    25%             5.100000          2.800000           1.600000   
    50%             5.800000          3.000000           4.350000   
    75%             6.400000          3.300000           5.100000   
    max             7.900000          4.400000           6.900000   
    
           petal width (cm)  
    count        150.000000  
    mean           1.199333  
    std            0.762238  
    min            0.100000  
    25%            0.300000  
    50%            1.300000  
    75%            1.800000  
    max            2.500000  
    sepal length (cm)    0
    sepal width (cm)     0
    petal length (cm)    0
    petal width (cm)     0
    Species              0
    dtype: int64
    1
         sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  \
    129                7.2               3.0                5.8               1.6   
    130                7.4               2.8                6.1               1.9   
    124                6.7               3.3                5.7               2.1   
    50                 7.0               3.2                4.7               1.4   
    26                 5.0               3.4                1.6               0.4   
    
            Species  
    129   Virginica  
    130   Virginica  
    124   Virginica  
    50   Versicolor  
    26       Setosa  
    sepal length (cm)    35
    sepal width (cm)     23
    petal length (cm)    43
    petal width (cm)     22
    Species               3
    dtype: int64
    Species
    Setosa        50
    Versicolor    50
    Virginica     50
    Name: count, dtype: int64
    

.. code:: ipython3

    for col in iris.feature_names:
        print(col)
        print("Mean :", df[col].mean())
        print("Median :", df[col].median())
        print("Mode :", df[col].mode()[0])
        print("Variance :", df[col].var())
        print("Standard Deviation :", df[col].std())
        print("Range :", df[col].max()-df[col].min())
        print("Skewness :", df[col].skew())
        print("Kurtosis :", df[col].kurt())
        print("-"*40)


.. parsed-literal::

    sepal length (cm)
    Mean : 5.843333333333334
    Median : 5.8
    Mode : 5.0
    Variance : 0.6856935123042505
    Standard Deviation : 0.8280661279778629
    Range : 3.6000000000000005
    Skewness : 0.3149109566369728
    Kurtosis : -0.5520640413156395
    ----------------------------------------
    sepal width (cm)
    Mean : 3.0573333333333337
    Median : 3.0
    Mode : 3.0
    Variance : 0.1899794183445188
    Standard Deviation : 0.435866284936698
    Range : 2.4000000000000004
    Skewness : 0.31896566471359966
    Kurtosis : 0.2282490424681929
    ----------------------------------------
    petal length (cm)
    Mean : 3.7580000000000005
    Median : 4.35
    Mode : 1.4
    Variance : 3.1162778523489942
    Standard Deviation : 1.7652982332594667
    Range : 5.9
    Skewness : -0.27488417975101276
    Kurtosis : -1.4021034155217518
    ----------------------------------------
    petal width (cm)
    Mean : 1.1993333333333336
    Median : 1.3
    Mode : 0.2
    Variance : 0.5810062639821029
    Standard Deviation : 0.7622376689603465
    Range : 2.4
    Skewness : -0.10296674764898116
    Kurtosis : -1.340603996612646
    ----------------------------------------
    

.. code:: ipython3

    group = df.groupby("Species")
    
    print(group.mean())
    print(group.std())
    
    print(
    group['petal length (cm)'].agg(['mean','median','std','skew'])
    )


.. parsed-literal::

                sepal length (cm)  sepal width (cm)  petal length (cm)  \
    Species                                                              
    Setosa                  5.006             3.428              1.462   
    Versicolor              5.936             2.770              4.260   
    Virginica               6.588             2.974              5.552   
    
                petal width (cm)  
    Species                       
    Setosa                 0.246  
    Versicolor             1.326  
    Virginica              2.026  
                sepal length (cm)  sepal width (cm)  petal length (cm)  \
    Species                                                              
    Setosa               0.352490          0.379064           0.173664   
    Versicolor           0.516171          0.313798           0.469911   
    Virginica            0.635880          0.322497           0.551895   
    
                petal width (cm)  
    Species                       
    Setosa              0.105386  
    Versicolor          0.197753  
    Virginica           0.274650  
                 mean  median       std      skew
    Species                                      
    Setosa      1.462    1.50  0.173664  0.106394
    Versicolor  4.260    4.35  0.469911 -0.606508
    Virginica   5.552    5.55  0.551895  0.549445
    

.. code:: ipython3

    import seaborn as sns
    import matplotlib.pyplot as plt
    
    df = sns.load_dataset("iris")

.. code:: ipython3

    plt.figure(figsize=(6,4))
    sns.histplot(df["sepal_length"], bins=15, kde=True)
    plt.title("Histogram of Sepal Length - 24EU02035")
    plt.show()



.. image:: output_5_0.png


.. code:: ipython3

    plt.figure(figsize=(6,4))
    sns.countplot(x="species", data=df)
    plt.title("Species Count - 24EU02035")
    plt.show()



.. image:: output_6_0.png


.. code:: ipython3

    plt.figure(figsize=(6,4))
    sns.boxplot(y="petal_length", data=df)
    plt.title("Box Plot of Petal Length - 24EU02035")
    plt.show()



.. image:: output_7_0.png


.. code:: ipython3

    plt.figure(figsize=(6,4))
    sns.kdeplot(df["sepal_width"], fill=True)
    plt.title("Density Plot of Sepal Width - 24EU02035")
    plt.show()



.. image:: output_8_0.png


.. code:: ipython3

    plt.figure(figsize=(6,4))
    sns.scatterplot(x="sepal_length",
                    y="petal_length",
                    hue="species",
                    data=df)
    
    plt.title("Sepal Length vs Petal Length - 24EU02035")
    plt.show()



.. image:: output_9_0.png


.. code:: ipython3

    sns.pairplot(df, hue="species")
    plt.suptitle("Pair Plot - 24EU02035", y=1.02)
    plt.show()



.. image:: output_10_0.png


.. code:: ipython3

    plt.figure(figsize=(7,5))
    sns.heatmap(df.corr(numeric_only=True),
                annot=True,
                cmap="coolwarm")
    
    plt.title("Correlation Heatmap - 24EU02035")
    plt.show()



.. image:: output_11_0.png


.. code:: ipython3

    plt.figure(figsize=(6,4))
    sns.boxplot(x="species",
                y="petal_length",
                data=df)
    
    plt.title("Petal Length by Species - 24EU02035")
    plt.show()



.. image:: output_12_0.png


.. code:: ipython3

    plt.figure(figsize=(6,4))
    sns.violinplot(x="species",
                   y="sepal_width",
                   data=df)
    
    plt.title("Sepal Width by Species - 24EU02035")
    plt.show()



.. image:: output_13_0.png


.. code:: ipython3

    plt.figure(figsize=(6,4))
    sns.barplot(x="species",
                y="petal_width",
                data=df)
    
    plt.title("Average Petal Width by Species - 24EU02035")
    plt.show()



.. image:: output_14_0.png


.. code:: ipython3

    plt.figure(figsize=(6,4))
    sns.barplot(x="species",
                y="petal_width",
                data=df)
    
    plt.title("Average Petal Width by Species - 24EU02035")
    plt.show()



.. image:: output_15_0.png


.. code:: ipython3

    import pandas as pd
    
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/sonar.csv"
    
    df = pd.read_csv(url, header=None)
    
    df.rename(columns={60:"Target"}, inplace=True)
    
    print(df.head())
    print(df.shape)
    print(df.dtypes)
    print(df.describe())
    print(df.isnull().sum())
    print(df.duplicated().sum())
    print(df["Target"].value_counts())


.. parsed-literal::

            0       1       2       3       4       5       6       7       8  \
    0  0.0200  0.0371  0.0428  0.0207  0.0954  0.0986  0.1539  0.1601  0.3109   
    1  0.0453  0.0523  0.0843  0.0689  0.1183  0.2583  0.2156  0.3481  0.3337   
    2  0.0262  0.0582  0.1099  0.1083  0.0974  0.2280  0.2431  0.3771  0.5598   
    3  0.0100  0.0171  0.0623  0.0205  0.0205  0.0368  0.1098  0.1276  0.0598   
    4  0.0762  0.0666  0.0481  0.0394  0.0590  0.0649  0.1209  0.2467  0.3564   
    
            9  ...      51      52      53      54      55      56      57  \
    0  0.2111  ...  0.0027  0.0065  0.0159  0.0072  0.0167  0.0180  0.0084   
    1  0.2872  ...  0.0084  0.0089  0.0048  0.0094  0.0191  0.0140  0.0049   
    2  0.6194  ...  0.0232  0.0166  0.0095  0.0180  0.0244  0.0316  0.0164   
    3  0.1264  ...  0.0121  0.0036  0.0150  0.0085  0.0073  0.0050  0.0044   
    4  0.4459  ...  0.0031  0.0054  0.0105  0.0110  0.0015  0.0072  0.0048   
    
           58      59  Target  
    0  0.0090  0.0032       R  
    1  0.0052  0.0044       R  
    2  0.0095  0.0078       R  
    3  0.0040  0.0117       R  
    4  0.0107  0.0094       R  
    
    [5 rows x 61 columns]
    (208, 61)
    0         float64
    1         float64
    2         float64
    3         float64
    4         float64
               ...   
    56        float64
    57        float64
    58        float64
    59        float64
    Target     object
    Length: 61, dtype: object
                   0           1           2           3           4           5   \
    count  208.000000  208.000000  208.000000  208.000000  208.000000  208.000000   
    mean     0.029164    0.038437    0.043832    0.053892    0.075202    0.104570   
    std      0.022991    0.032960    0.038428    0.046528    0.055552    0.059105   
    min      0.001500    0.000600    0.001500    0.005800    0.006700    0.010200   
    25%      0.013350    0.016450    0.018950    0.024375    0.038050    0.067025   
    50%      0.022800    0.030800    0.034300    0.044050    0.062500    0.092150   
    75%      0.035550    0.047950    0.057950    0.064500    0.100275    0.134125   
    max      0.137100    0.233900    0.305900    0.426400    0.401000    0.382300   
    
                   6           7           8           9   ...          50  \
    count  208.000000  208.000000  208.000000  208.000000  ...  208.000000   
    mean     0.121747    0.134799    0.178003    0.208259  ...    0.016069   
    std      0.061788    0.085152    0.118387    0.134416  ...    0.012008   
    min      0.003300    0.005500    0.007500    0.011300  ...    0.000000   
    25%      0.080900    0.080425    0.097025    0.111275  ...    0.008425   
    50%      0.106950    0.112100    0.152250    0.182400  ...    0.013900   
    75%      0.154000    0.169600    0.233425    0.268700  ...    0.020825   
    max      0.372900    0.459000    0.682800    0.710600  ...    0.100400   
    
                   51          52          53          54          55          56  \
    count  208.000000  208.000000  208.000000  208.000000  208.000000  208.000000   
    mean     0.013420    0.010709    0.010941    0.009290    0.008222    0.007820   
    std      0.009634    0.007060    0.007301    0.007088    0.005736    0.005785   
    min      0.000800    0.000500    0.001000    0.000600    0.000400    0.000300   
    25%      0.007275    0.005075    0.005375    0.004150    0.004400    0.003700   
    50%      0.011400    0.009550    0.009300    0.007500    0.006850    0.005950   
    75%      0.016725    0.014900    0.014500    0.012100    0.010575    0.010425   
    max      0.070900    0.039000    0.035200    0.044700    0.039400    0.035500   
    
                   57          58          59  
    count  208.000000  208.000000  208.000000  
    mean     0.007949    0.007941    0.006507  
    std      0.006470    0.006181    0.005031  
    min      0.000300    0.000100    0.000600  
    25%      0.003600    0.003675    0.003100  
    50%      0.005800    0.006400    0.005300  
    75%      0.010350    0.010325    0.008525  
    max      0.044000    0.036400    0.043900  
    
    [8 rows x 60 columns]
    0         0
    1         0
    2         0
    3         0
    4         0
             ..
    56        0
    57        0
    58        0
    59        0
    Target    0
    Length: 61, dtype: int64
    0
    Target
    M    111
    R     97
    Name: count, dtype: int64
    

.. code:: ipython3

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    corr = df.iloc[:, :-1].corr()
    
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap="coolwarm")
    
    plt.title("Correlation Matrix - 24EU02035")
    plt.show()



.. image:: output_17_0.png


.. code:: ipython3

    print(df.corr(numeric_only=True)[0].sort_values(ascending=False))


.. parsed-literal::

    0     1.000000
    1     0.735896
    2     0.571537
    3     0.491438
    57    0.368132
    58    0.357116
    7     0.355523
    51    0.355299
    8     0.353420
    59    0.347078
    4     0.344797
    10    0.344058
    53    0.322299
    45    0.319354
    9     0.318276
    56    0.313725
    54    0.312067
    52    0.311729
    14    0.304878
    44    0.279968
    49    0.269287
    6     0.260815
    13    0.256278
    50    0.254450
    48    0.247560
    15    0.239079
    5     0.238921
    46    0.230343
    55    0.220642
    41    0.213592
    11    0.210861
    12    0.210722
    37    0.209873
    38    0.208371
    42    0.206057
    47    0.203234
    43    0.157949
    19    0.156760
    16    0.137845
    40    0.127313
    36    0.119565
    20    0.117663
    39    0.099993
    34    0.098118
    35    0.080722
    18    0.055227
    17    0.041817
    33    0.031319
    31   -0.030444
    32   -0.031939
    30   -0.048370
    21   -0.056973
    29   -0.077430
    22   -0.163426
    28   -0.199099
    23   -0.218093
    27   -0.224340
    24   -0.295683
    26   -0.341703
    25   -0.342865
    Name: 0, dtype: float64
    

