%%
load('gluthdataout.mat');

%% model free
Nsubj = length(data);
Bmat = [];
for s = 1:Nsubj
    dataIn = data(s);
    dataIn.X = cell2mat(dataIn.X);
    condmask = dataIn.y ~= 3;
    
    x1 = [dataIn.X(1,condmask) - dataIn.X(2,condmask)]';
    x2 = dataIn.X(3, condmask)';
    y = 2-dataIn.y(condmask);
    b = glmfit([x1 x2], y, 'binomial', 'link', 'probit');
    yfit = glmval(b, [x1 x2], 'probit', 'size', 1);
    plot(x1, y, 'o', x1, yfit, '-')
    Bmat(:,s) = b;
end

%%
h = figure;
subplot(2,1,1);
bar(Bmat(2,:));
subplot(2,1,2);
bar(Bmat(3,:));
ylim([-1,1]);
%% MLE fitting, logit
numInit = 30;
LB = [0 -Inf]; % sigma, omega
UB = [Inf Inf];
PLB = [0 -2];
PUB = [100 2];
OPTIONS.MaxFunEvals = 2000;
OPTIONS = optimset(OPTIONS,'Display', 'iter-detailed', 'MaxIter', 10000, 'Algorithm', 'interior-point','OutputFcn',@outfun); %'active-set',
% OPTIONS=optimset('MaxIter',10000,'Algorithm','interior-point','Display','on');
disp(sprintf('TolX: %g   TolFun: %g',OPTIONS.TolX,OPTIONS.TolFun));
Nsubj = numel(data);
MLEoutII = [];
for s = 1:Nsubj
    dataIn = data(s);
    f = @(x)(LLFUN(x,dataIn));
    X0 = PLB + rand(1,2).*(PUB - PLB);
    count = 0;
    XBest = [];
    for i = 1:numInit
        [X, tMin] = fmincon(f,X0,[],[],[],[],LB,UB,[],OPTIONS);
        if ~isnan(tMin) && (count == 0 || (tMin < ll_min))
            XBest = X;
            ll_min = tMin;
            count = count + 1;
        end
    end
    MLEoutII(s).parh = XBest;
end

%% visualize fitted data
Nsubj = numel(MLEoutII);
sigma = [];
omega = [];
for subj = 1:Nsubj
   sigma(subj) =  MLEoutII(subj).parh(1);
   omega(subj) = MLEoutII(subj).parh(2);
end
h = figure;
subplot(2,1,1);
bar(sigma);
subplot(2,1,2);
bar(omega);


%% negative loglikelihood function, logit
function nLL = LLFUN(params, dataIn)
sigma = params(1);
w = params(2);
DN = @(x)(x./(sigma + sum(w*x)));
V = cellfun(DN, dataIn.X, 'UniformOutput', false); % DN transformed value
cntrstMat = {[-1 1 0; -1 0 1], [1 -1 0;0 -1 1], [1 0 -1; 0 1 -1]}; % contrast matrix for multinomial logistic
Mi = {cntrstMat{dataIn.y}}; % different contrast based on choice
celltime = @(x,y)(x*y);
v = cellfun(celltime,Mi,V,'UniformOutput',false); % pivoted linear predictor in multinomial logistic
sum_expv = cellfun(@(x)(sum(exp(x))), v, 'UniformOutput', true); % summed exp(x) in dinominator
P = 1./(1+sum_expv);
logP=log(P);
nLL=-sum(logP);
end

function stop = outfun(x,~,state) %display parameter values during estimation
stop = false;
if state == 'iter'
    disp([' x = ',num2str(x)]);
    save('iteroutput')
end
end