function info = jacobi_iteration(A,varargin) 
% uncoupled algorithms based on als and asvd, i.e., jacobi iteration

N = ndims(A);
sizeA = A.size;

params = inputParser;
params.addParameter('method', 'asvd', @(x) ismember(x,{'als','asvd'}));
params.addParameter('tol',1.0e-4);
params.addParameter('maxiters',50);
params.addParameter('init','random', @(x) (iscell(x) || ismember(x,{'random','t','st'})));
params.parse(varargin{:});

method = params.Results.method;
tol = params.Results.tol;
maxiters = params.Results.maxiters;
init = params.Results.init;

if iscell(init)
    v = init;
elseif strcmp(init,'random')
    v = cell(N,1);
    for n = 1:N
        v{n} = rand(sizeA(n),1);
        v{n} = v{n}/norm(v{n});
    end
elseif strcmp(init,'t')
    T = hosvd(A,1.0,'ranks',ones(1,N),'Sequential',false);
    v = T.U;
elseif strcmp(init,'st')
    T = hosvd(A,1.0,'ranks',ones(1,N));
    v  = T.U;
end

res_v = [];

if strcmp(method,'als')
    for i = 1:maxiters
        v_new = cell(N,1);
        for n = 1:N
            mode = [1:n-1 n+1:N];
            v_inter = ttv(A,v(mode),mode); v_inter = v_inter.data;
            v_new{n} = v_inter/norm(v_inter);
        end

        v = v_new;
        lambda = ttv(A,v,1:N);
        res = residual(A,lambda,v);

        res_v(i) = res;
        fprintf("%d-th jals iteration: lambda is %f, residual is %7.1e\n",i,lambda,res);

        if res <= tol
            break;
        end
    end

elseif strcmp(method,'asvd')
    if mod(N,2) == 1
        for i = 1:maxiters
            v_new = cell(N,1);
            for n = 1:2:N-2
                mode = [1:n-1 n+2:N];
                mat_inter = ttv(A,v(mode),mode); mat_inter = mat_inter.data;
                [u1,~,u2] = svds(mat_inter,1);
                v_new{n} = u1; v_new{n+1} = u2;
            end
            mode = 1:N-1;
            v_inter = ttv(A,v(mode),mode); v_inter = v_inter.data;
            v_new{N} = v_inter/norm(v_inter);

            v = v_new;
            lambda = ttv(A,v,1:N);
            res = residual(A,lambda,v);
    
            res_v(i) = res;
            fprintf("%d-th jasvd iteration: lambda is %f, residual is %7.1e\n",i,lambda,res);
    
            if res <= tol
                break;
            end
        end

    elseif mod(N,2) == 0
        for i = 1:maxiters
            v_new = cell(N,1);
            for n = 1:2:N
                mode = [1:n-1 n+2:N];
                mat_inter = ttv(A,v(mode),mode); mat_inter = mat_inter.data;
                [u1,~,u2] = svds(mat_inter,1);
                v_new{n} = u1; v_new{n+1} = u2;
            end

            v = v_new;
            lambda = ttv(A,v,1:N);
            res = residual(A,lambda,v);
    
            res_v(i) = res;
            fprintf("%d-th jasvd iteration: lambda is %f, residual is %7.1e\n",i,lambda,res);
    
            if res <= tol
                break;
            end
        end
    end
end

info.method = method;
info.lambda = lambda;
info.v = v;
info.iteration = i;
info.residual = res_v;

end

function res = residual(A,lambda,v)
% used to calculate the residual

N = ndims(A);
sizeA = A.size;

J = zeros(sum(sizeA),sum(sizeA));
w = zeros(sum(sizeA),1);
for i = 1:N-1
    if i == 1
        s1 = 0;
    else
        s1 = sum(sizeA(1:i-1));
    end
    index1 = s1+1:s1+sizeA(i);
    for j = i+1:N
        mode = [1:i-1 i+1:j-1 j+1:N];
        J_inter = ttv(A,v(mode),mode); J_inter = J_inter.data;

        s2 = sum(sizeA(1:j-1));
        index2 = s2+1:s2+sizeA(j);
        J(index1,index2) = J_inter/(N-1);
        J(index2,index1) = J_inter'/(N-1);
    end
    w(index1) = v{i};
end
s1 = sum(sizeA(1:N-1));
index1 = s1+1:s1+sizeA(N);
w(index1) = v{N};
w = w/norm(w);

w_inter = J*w; rho = w'*w_inter;
res = norm(w_inter - rho*w)/(norm(J,'fro')+abs(lambda));
% res = norm(w_inter - rho*w)/(norm(J,'fro'));

end