function info = hoscf(A,varargin)

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

info = scf_iteration(A,'init',v,'tol',tol,'maxiters',maxiters);

end

function info = scf_iteration(A,varargin)
% uncoupled algorithm based on scf iteration 

N = ndims(A);
sizeA = A.size;

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
J = zeros(sum(sizeA),sum(sizeA));
w = zeros(sum(sizeA),1);
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

    w_inter = J*w; rho = w'*w_inter;
    res = norm(w_inter - rho*w)/(norm(J,'fro')+abs(lambda));
%     res = norm(w_inter - rho*w)/(norm(J,'fro'));
    res_v(i) = res;
    fprintf("%d-th scf iteration: lambda is %f, residual is %7.1e\n",i,lambda,res);

    if res <= tol
        break;
    else
%         [w,lambda] = eigs(J,1);
        [w,lambda,~] = svds(J,1);
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