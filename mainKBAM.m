% 
% (c) 2019 Naoki Masuyama
% 
% These are the codes of Kernel Bayesian Adaptive Resonance Theory (KBA)
% proposed in "N. Masuyama, C. L. Loo, and F. Dawood, Kernel Bayesian 
% ART and ARTMAP, Neural Networks, vol. 98, pp. 76-86, November 2017."
% 
% Please contact "masuyama@cs.osakafu-u.ac.jp" if you have any problem.
% 



load iris_dataset
Data = irisInputs;
Label = irisTargets;

% scaling [0,1]
Data = normalize(Data,'range');

% Randamization
ran = randperm(size(Data,1));
Data = Data(ran,:);
Label = Label(ran,:);

% Traingin data
trainD = Data(1:15,:);
trainL = Label(1:15,:);
% Testing data
testD = Data(16:150,:);
testL = Label(16:150,:);


% Parameters of KBAM
KBAMnet.weight    = [];          % Mean of cluster
KBAMnet.mapField  = [];          % Map
KBAMnet.numClusters = 0;         % Number of clusters
KBAMnet.Pmin = 0.55;             % Probability Threshold
KBAMnet.bias = 1e-6;             % Bias for Vigilance parameter
KBAMnet.maxNumClusters = inf;    % Maximum number of clusters
KBAMnet.ClusterAttribution = []; % Cluster attribution for each input
KBAMnet.CountCluster = 0;        % Counter for each cluster

KBAMnet.maxCIM = 0.15;  % Vmax
KBAMnet.kbrSig = 0.1;  % \sigma_kbr
KBAMnet.cimSig = 0.1; % \sigma_cim


% Train Network
KBAMnet = KBAM_train(KBAMnet, trainD, trainL);

% Test 
[acc, estLabels, MapPostPr] = KBAM_test(KBAMnet, testD, testL);

disp(['Accuracy: ',num2str(acc)]);
disp(['# of Clusters: ',num2str(size(KBAMnet.weight,1))]);



