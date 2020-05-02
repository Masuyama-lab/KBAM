function [acc, estLabels, MapPostPr] = KBAM_test(net, patterns, labels)

[numSamples, ~] = size(patterns);
ClusterPostPr = nan(numSamples, net.numClusters);

% Parameters for Kernel Bayes Rule
paramKBR.Sig     = net.kbrSig;           % Kernel bandwidth  % Need to adjast Kernel bandwidth
paramKBR.numNode = net.numClusters;      % Number of Clusters
paramKBR.Eps     = 0.01/net.numClusters; % Scaling Factor
paramKBR.Delta   = 2*net.numClusters;    % Scaling Factor
paramKBR.gamma   = ones(numSamples,1) / numSamples;  % Scaling Factor

for k = 1:numSamples
    % Kernel Bayes Rule
    [KernelPo] = KernelBayesRule(patterns(k,:), net.weight, paramKBR);
    ClusterPostPr(k,:) = KernelPo'; % Cluster Posterior Probability
end

% Map Field Posterior Probability
MapPostPr = ClusterPostPr * net.mapField;
MapPostPr = MapPostPr ./ (sum(MapPostPr,2) * ones(1,size(MapPostPr,2)));

% Classify each of a list of patterns.
[~, estLabels] = max(MapPostPr,[],2);

% Compute Accuracy
acc = sum( estLabels == labels(:,1:size(estLabels, 2)) ) / length(labels);


end



% Kernel Bayes Rule
function [Po] = KernelBayesRule(pattern, weight, paramKBR)

input = pattern;
meanU = mean(input,1);

% Parameters for Kernel Bayes Rule
Sig = paramKBR.Sig;           % Kernel bandwidth
numNode = paramKBR.numNode;   % Number of Nodes
Eps = paramKBR.Eps;           % Scaling Factor
Delta = paramKBR.Delta;       % Scaling Factor
gamma = paramKBR.gamma;       % Scaling Factor

% Calculate Gram Matrix
Pr = ones(size(weight,1), size(weight,2)) / size(weight,1);  % Prior Probability
G_theta=Gramian(weight, weight, Sig);
G_Pr=Gramian(Pr, Pr, Sig);

m_hat = zeros(numNode,1);
tmp = zeros(size(Pr,1),size(input,1));
for i=1:size(Pr,1)
   for j=1:size(input,1)
       tmp(i,j) = gamma(j) * gaussian_kernel(Pr(i,:), input(j,:), Sig);
   end
   m_hat(i) = sum(tmp(i,:),2);
end

mu_hat = numNode \ (G_Pr + numNode * Eps * eye(numNode)) * m_hat;
Lambda = diag(mu_hat);
LG = Lambda * G_theta;
R = LG \ (LG^2 + Delta * eye(numNode)) * Lambda;
tmpPo = R * gaussian_kernel(weight, meanU, Sig);
tmpPo(tmpPo < 0) = 0;

Po = tmpPo / sum(tmpPo); % Posterior Probability
% posteriorMean = weight'*Po; % Estimated Mean

end

% Gram Matrix
function gram = Gramian(X1, X2, sig)
a=X1'; b=X2';
if (size(a,1) == 1)
  a = [a; zeros(1,size(a,2))];  
  b = [b; zeros(1,size(b,2))];  
end 
aa=sum(a.*a); bb=sum(b.*b); ab=a'*b;  
D = sqrt(repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);

gram = exp(-(D.^2 / (2 * sig.^2)));
end

% Gaussian Kernel
function g_kernel = gaussian_kernel(X, W, sig)

nrm = sum(bsxfun(@minus, X, W).^2, 2);
g_kernel = exp(-nrm/(2*sig^2));

end

