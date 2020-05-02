function net = KBAM_train(net, patterns, labels)



weight = net.weight;                 % Mean of cluster
mapField = net.mapField;             % map
numClusters = net.numClusters;       % Number of clusters
Pmin = net.Pmin;                     % Probability Threshold
bias = net.bias;                     % Bias for Vigilance parameter
maxNumClusters = net.maxNumClusters; % Maximum number of clusters
ClusterAttribution = net.ClusterAttribution;

maxCIM = net.maxCIM;                 % Vigilance Parameter by CIM
kbrSig = net.kbrSig;                 % Kernel Bandwidth for Kernel Bayes Rule
cimSig = net.cimSig;                 % Kernel Bandwidth for CIM



% Classify and learn on each sample.
numSamples = size(patterns,1);

    
for sampleNum = 1:numSamples
    
    % Get the current data sample.
    pattern = patterns(sampleNum,:);
    label = labels(sampleNum);
    
    
    % Kernel Bayes
    if sampleNum == 1 && isempty(weight) == 1
        KernelPo = nan(1, numClusters); % Due to there is no cluster yet
    else
        % Parameters for Kernel Bayes Rule
        paramKBR.Sig     = kbrSig;        % Kernel bandwidth  % Need to adjast Kernel bandwidth
        paramKBR.numNode = numClusters;      % Number of Clusters
        paramKBR.Eps     = 0.01/numClusters; % Scaling Factor
        paramKBR.Delta   = 2*numClusters;    % Scaling Factor
        paramKBR.gamma   = ones(size(pattern, 1),1) / size(pattern, 1);  % Scaling Factor
        paramKBR.prior   = sum(mapField,2) / sum(sum(mapField));    % Prior Probability
        
        % Kernel Bayes Rule
        [KernelPo, ~] = KernelBayesRule(pattern, weight, paramKBR); % return [PostPr, posteriorMean]
    end
    
    [~, sortedClusters] = sort(-KernelPo); % Normal Bayes Rule
    
    resonance = false;
    numSortedClusters = length(sortedClusters);
    currentSortedIndex = 1;
    
    
    % only for first cluster
    if numSortedClusters == 0 && isempty(weight) == 1
        % Add Cluster
        numClusters             = numClusters + 1;
        weight(numClusters,:)   = pattern;
        mapField(numClusters, label) = 1;
        ClusterAttribution(1, sampleNum) = 1;
        
        resonance = true;
    end
    
    
    patternsPerCluster = sum(mapField,2);
    currentCIM = maxCIM;
    
    while ~resonance
        
        bestCluster = sortedClusters(currentSortedIndex);
        newWeight = (patternsPerCluster(bestCluster)*weight(bestCluster,:) + pattern) / (patternsPerCluster(bestCluster)+1);
        
        % Calculate an error based on CIM
        bestCIM = CIM(pattern, newWeight, cimSig);
        
        % Vigilance Test
        if bestCIM <= currentCIM
            % Match Success
            ClusterAttribution(1, sampleNum) = bestCluster;
        else
            % Match Fail
            if(currentSortedIndex == numSortedClusters)  % Reached to maximum number of generated clusters
                if(currentSortedIndex == maxNumClusters)    % Reached to defined muximum number of clusters
                    ClusterAttribution(1, sampleNum) = -1;
                    fprintf('WARNING: The maximum number of clusters has been reached.\n');
                    resonance = true;
                else
                    % Add Cluster
                    numClusters             = numClusters + 1;
                    weight(numClusters,:)   = pattern;
                    mapField(numClusters, label) = 1;
                    
                    ClusterAttribution(1, sampleNum) = currentSortedIndex + 1;
                    
                    resonance = true;
                end
            else
                currentSortedIndex = currentSortedIndex + 1;    % Search another cluster orderd by sortedClusters
            end
            
        end % end Vigilance Test
        
        if resonance
            break;
        end
        
        newMap = mapField;
        if label>size(newMap,2)
            newMap(bestCluster,label) = 0;
        end
        
        newMap(bestCluster,label) = newMap(bestCluster,label) + 1;
        
        % Match Tranking
        if newMap(bestCluster,label) / sum(newMap(bestCluster,:)) >= Pmin
            % Success
            % Update Network Prameters
            weight(bestCluster,:) = newWeight;
            mapField = newMap;
            break;
        else
            % Fail
            % Update Vigilance Parameter
            bestCIM = CIM(pattern, weight(bestCluster,:), cimSig);
            currentCIM = bestCIM - bias;
            
            if(currentSortedIndex == numSortedClusters)  % Reached to maximum number of generated clusters
                if(numClusters == maxNumClusters)    % Reached to defined muximum number of clusters
                    fprintf('WARNING: The maximum number of clusters has been reached.\n');
                    resonance = true;
                else
                    % Add Cluster
                    numClusters             = numClusters + 1;
                    weight(numClusters,:)   = pattern;
                    mapField(numClusters, label) = 1;
                    ClusterAttribution(1, sampleNum) = currentSortedIndex + 1;
                    resonance = true;
                end
            else
                currentSortedIndex = currentSortedIndex + 1;    % Search another cluster orderd by sortedClusters
            end
            
        end
        
    end % end resonance
    
end % end numSample
    

net.weight = weight;
net.mapField = mapField;
net.numClusters = numClusters;
net.ClusterAttribution = ClusterAttribution;
net.maxCIM = maxCIM;

end





% Kernel Bayes Rule
function [Po, posteriorMean] = KernelBayesRule(pattern, weight, paramKBR)

input = pattern;
meanU = mean(input,1);

% Parameters for Kernel Bayes Rule
Sig = paramKBR.Sig;           % Kernel bandwidth
numNode = paramKBR.numNode;   % Number of Nodes
Eps = paramKBR.Eps;           % Scaling Factor
Delta = paramKBR.Delta;       % Scaling Factor
gamma = paramKBR.gamma;       % Scaling Factor
Pr = paramKBR.prior;          % Prior Probability


% Calculate Gram Matrix
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
posteriorMean = weight'*Po; % Estimated Mean

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


% Correntropy induced Metric (Gaussian Kernel based)
function cim = CIM(X,Y,sig)
% X : 1 x n
% Y : m x n
[n, att] = size(Y);
g_Kernel = zeros(n, att);

for i = 1:att
    g_Kernel(:,i) = GaussKernel(X(i)-Y(:,i), sig);
end

ret0 = GaussKernel(0, sig);
ret1 = mean(g_Kernel, 2);

cim = sqrt(ret0 - ret1)';
end

% Gaussian Kernel
function g_kernel = GaussKernel(sub, sig)
g_kernel = exp(-sub.^2/(2*sig^2));
% g_kernel = 1/(sqrt(2*pi)*sig) * exp(-sub.^2/(2*sig^2));
end

% Gaussian Kernel
function g_kernel = gaussian_kernel(X, W, sig)
nrm = sum(bsxfun(@minus, X, W).^2, 2);
g_kernel = exp(-nrm/(2*sig^2));
end



