function info = ihoscf(A,varargin)

N = ndims(A);
sizeA = A.size;

if mod(N,2) == 1
    A = A.data;
    A = reshape(A,[1,sizeA]);
    A = tensor(A);
    N = ndims(A);
    sizeA = A.size;
end

params = inputParser;
params.addParameter('tol',1.0e-4);
params.addParameter('maxiters',50);
params.addParameter('init','random', @(x) (iscell(x) || ismember(x,{'random','t','st'})));
params.parse(varargin{:});

tol = params.Results.tol;
maxiters = params.Results.maxiters;
init = params.Results.init;

if iscell(init)
    v = init;
elseif strcmp(init,'random')
    v = cell(N,1);
    for n = 1:N
        v_inter = rand(sizeA(n),1);
        v{n} = v_inter/norm(v_inter);
    end
elseif strcmp(init,'t')
    T = hosvd(A,1.0,'ranks',ones(1,N),'Sequential',false);
    v = T.U;
elseif strcmp(init,'st')
    T = hosvd(A,1.0,'ranks',ones(1,N));
    v  = T.U;
end

info = ascf_iteration(A,'init',v,'tol',tol,'maxiters',maxiters);

end

function info = ascf_iteration(A,varargin)
% uncoupled algorithm based on an accelerated scf iteration 

N = ndims(A);
sizeA = A.size;
I = sum(sizeA);

params = inputParser;
params.addParameter('tol',1.0e-4);
params.addParameter('maxiters',50);
params.addParameter('init','random', @(x) (iscell(x) || ismember(x,{'random','t','st'})));
params.parse(varargin{:});

tol = params.Results.tol;
maxiters = params.Results.maxiters;
init = params.Results.init;

if iscell(init)
    v = init;
    lambda = ttv(A,v,1:N);
elseif strcmp(init,'random')
    v = cell(N,1);
    for n = 1:N
        v_inter = rand(sizeA(n),1);
        v{n} = v_inter/norm(v_inter);
    end
    lambda = ttv(A,v,1:N);
elseif strcmp(init,'t')
    T = hosvd(A,1.0,'ranks',ones(1,N),'Sequential',false);
    v = T.U;
    lambda = ttv(A,v,1:N);
elseif strcmp(init,'st')
    T = hosvd(A,1.0,'ranks',ones(1,N));
    v  = T.U;
    lambda = ttv(A,v,1:N);
end

res_v = [];
w = zeros(I,1);
for n = 1:N
    if n == 1
        s1 = 0;
    else
        s1 = sum(sizeA(n-1));
    end
    index1 = s1+1:s1+sizeA(n);
    w(index1) = v{n};
end
w = w/norm(w);

for i = 1:maxiters
    % intermediate matrix J
    J = coefficient(A,v);

    w_inter = J*w; rho = w'*w_inter;
    res = norm(w_inter - rho*w)/(norm(J,'fro')+abs(lambda));
%     res = norm(w_inter - rho*w)/(norm(J,'fro'));
    res_v(i) = res;
    fprintf("%d-th ascf iteration: lambda is %f, residual is %7.1e\n",i,lambda,res);

    if res <= tol
        break;
    else
        w_new = (J-rho*eye(I))\w; w_new = w_new/norm(w_new);
        v_new = v;
    
        for n = 1:N
            if n == 1
                s1 = 0;
            else
                s1 = sum(sizeA(1:n-1));
            end
            index1 = s1+1:s1+sizeA(n);
            v_inter = w_new(index1); 
            v_new{n} = v_inter/norm(v_inter);
        end
        
        mode = 1:N;
        lambda_new = ttv(A,v_new,mode); lambda_new = abs(lambda_new);

        if lambda_new > lambda
            J = coefficient(A,v_new);
        end

        [w,lambda] = eigs(J,1);
        for n = 1:N
            if n == 1
                s1 = 0;
            else
                s1 = sum(sizeA(1:n-1));
            end
            index1 = s1+1:s1+sizeA(n);
            v_inter = w(index1); 
            v{n} = v_inter/norm(v_inter);
        end
        lambda = ttv(A,v,1:N);
    end

end

info.lambda = lambda;
info.v = v;
info.iteration = i;
info.residual = res_v;

end

function J = coefficient(A,v)

% construct the intermediate matrix J

N = ndims(A);
sizeA = A.size;
I = sum(sizeA);

J = zeros([I,I]);

for n = 1:N-1
    if n == 1
        s1 = 0;
    else
        s1 = sum(sizeA(1:n-1));
    end
    index1 = s1+1:s1+sizeA(n);
    for m = n+1:N
        mode = [1:n-1 n+1:m-1 m+1:N];
        J_inter = ttv(A,v(mode),mode); J_inter = J_inter.data;

        s2 = sum(sizeA(1:m-1));
        index2 = s2+1:s2+sizeA(m);
        J(index1,index2) = J_inter/(N-1);
        J(index2,index1) = J_inter'/(N-1);
    end
end

end