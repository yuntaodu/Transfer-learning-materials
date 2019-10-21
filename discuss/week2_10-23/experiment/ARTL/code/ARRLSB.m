function [Acc,Cls,Alpha] = ARRLSB(Xs,Xt,Ys,Yt,options)
%% Mingsheng Long. Adaptation Regularization: A General Framework for Transfer Learning. TKDE 2012.

%% Load algorithm options
addpath(genpath('../liblinear/matlab'));

if nargin < 5
    error('Algorithm parameters should be set!');
end
if ~isfield(options,'p')
    options.p = 10;
end
if ~isfield(options,'sigma')
    options.sigma = 0.1;
end
if ~isfield(options,'lambda')
    options.lambda = 1.0;
end
if ~isfield(options,'gamma')
    options.gamma = 1.0;
end
if ~isfield(options,'ker')
    options.ker = 'linear';
end
if ~isfield(options,'data')
    options.data = 'default';
end
if ~isfield(options, 'Mu') || options.Mu < 0 || options.Mu > 1
    options.Mu = 0.1;
end
p = options.p;
sigma = options.sigma;
lambda = options.lambda;
gamma = options.gamma;
ker = options.ker;
data = options.data;
Mu = options.Mu;
fprintf('Algorithm ARRLSB started...\n');
fprintf('data=%s  p=%d  sigma=%f  lambda=%f  gamma=%f\n',data,p,sigma,lambda,gamma);

%% Set predefined variables
X = [Xs,Xt];
Y = [Ys;Yt];
n = size(Xs,2);
m = size(Xt,2);
nm = n+m;
E = diag(sparse([ones(n,1);zeros(m,1)]));
YY = [];
for c = reshape(unique(Y),1,length(unique(Y)))
    YY = [YY,Y==c];
end
[~,Y] = max(YY,[],2);

%% Data normalization
X = X*diag(sparse(1./sqrt(sum(X.^2))));

%% Construct graph Laplacian
manifold.k = options.p;
manifold.Metric = 'Cosine';
manifold.NeighborMode = 'KNN';
manifold.WeightMode = 'Cosine';
W = graph2(X',manifold);
Dw = diag(sparse(sqrt(1./sum(W))));
L = speye(nm)-Dw*W*Dw;

%% Construct MMD matrix
if ~isfield(options,'Yt0')
    model = train(Y(1:n),sparse(X(:,1:n)'),'-s 0 -c 1 -q 1');
    [Cls,~] = predict(Y(n+1:end),sparse(X(:,n+1:end)'),model);
else
    Cls = options.Yt0;
end
e = [1/n*ones(n,1);-1/m*ones(m,1)];
M = e*e'*length(unique(Y(1:n)))*Mu; % add Mu
for c = reshape(unique(Y(1:n)),1,length(unique(Y(1:n))))
    e = zeros(n+m,1);
    e(Y(1:n)==c) = 1/length(find(Y(1:n)==c));
    e(n+find(Cls==c)) = -1/length(find(Cls==c));
    e(isinf(e)) = 0;
    M = M + e*e'*(1-Mu); % add (1-Mu)
end
M = M/norm(M,'fro');

%% Adaptation Regularization based Transfer Learning: ARRLS
K = kernel(ker,X,sqrt(sum(sum(X.^2).^0.5)/nm));
Alpha = ((E+lambda*M+gamma*L)*K+sigma*speye(nm,nm))\(E*YY);
F = K*Alpha;
[~,Cls] = max(F,[],2);

%% Compute accuracy
Acc = numel(find(Cls(n+1:end)==Y(n+1:end)))/m;
Cls = Cls(n+1:end);
fprintf('>>Acc=%f\n',Acc);

fprintf('Algorithm ARRLSB terminated!!!\n\n');

end
