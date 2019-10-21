% With Fast Computation of the RBF kernel matrix
% To speed up the computation, we exploit a decomposition of the Euclidean distance (norm)
%
% Inputs:
%       ker:    'linear','rbf'
%       X:      data matrix with training samples in columns and features in rows
%       sigma:  width of the RBF kernel
% Output:
%       K: kernel matrix
%
% Gustavo Camps-Valls
% 2006(c)
% Jordi (jordi@uv.es), 2007
% 2007-11: if/then -> switch, and fixed RBF kernel
% Modified by Mingsheng Long
% 2013(c)
% Mingsheng Long (longmingsheng@gmail.com), 2013

function K = kernel(ker,X,sigma)

switch ker
    case 'linear'
        
        K = X' * X;

    case 'rbf'

        n1sq = sum(X.^2,1);
        n1 = size(X,2);
        
        D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
        K = exp(-D/(2*sigma^2));        

    case 'sam'
            
        D = X'*X;
        K = exp(-acos(D).^2/(2*sigma^2));

    otherwise
        error(['Unsupported kernel ' ker])
end