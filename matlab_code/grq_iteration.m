function info = grq_iteration(A,varargin)
% uncoupled algorithm based on grq iteration (grqi)

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
b = zeros(sum(sizeA),1);
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
        % diagonal blocks
        J_inter = -lambda*eye(sizeA(n));
        J(index1,index1) = J_inter;
        % non-diagonal blocks
        for m = n+1:N
            mode = [1:n-1 n+1:m-1 m+1:N];
            J_inter = ttv(A,v(mode),mode); J_inter = J_inter.data;

            s2 = sum(sizeA(1:m-1));
            index2 = s2+1:s2+sizeA(m);
            J(index1,index2) = J_inter;
            J(index2,index1) = J_inter';
        end
        % right-hand side
        mode = [1:n-1 n+1:N];
        b_inter = ttv(A,v(mode),mode); b_inter = b_inter.data;
        b(index1) = b_inter;
    end
    s1 = sum(sizeA(1:N-1));
    index1 = s1+1:s1+sizeA(N);
    J(index1,index1) = -lambda*eye(sizeA(N));

    mode = 1:N-1;
    b_inter = ttv(A,v(mode),mode); b_inter = b_inter.data;
    b(index1) = b_inter;

    res = residual(A,lambda,v);
    res_v(i) = res;
    fprintf("%d-th grq iteration: lambda is %f, residual is %7.1e\n",i,lambda,res);

    if res <= tol
        break;
    else
        w = J\b;
        w = w*(N-2);
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