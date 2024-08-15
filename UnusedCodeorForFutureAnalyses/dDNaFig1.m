function [probs, Ovlps, CVs] = dDNaFig1(pars, dat, num_samples) 
%% % Bo Shen, NYU, 2024
% distributional divisive normalization
% type a: inputs can be negative, assume inputs noise as fully independent
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end

eta = pars(1); % late noise standard deviation
scl = pars(2); % scaling for early noise
w = pars(3); % weight of normalization
M = pars(4); % baseline normalization
K = 75;
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3'});
Ntrl = size(dat,1);
Numerator = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
        stds = gpuArray(data.(['sdV', num2str(ci)])')*scl;
        Numerator(ci,:,:) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    else
        values = data.(['V',num2str(ci)])';
        stds = data.(['sdV', num2str(ci)])'*scl;
        Numerator(ci,:,:) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    end
end
Denominator1 = [];
Denominator2 = [];
Denominator3 = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
        stds = gpuArray(data.(['sdV', num2str(ci)])')*scl;
        Denominator1(ci,:,:) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
        Denominator2(ci,:,:) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
        Denominator3(ci,:,:) = gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    else
        values = data.(['V',num2str(ci)])';
        stds = data.(['sdV', num2str(ci)])'*scl;
        Denominator1(ci,:,:) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
        Denominator2(ci,:,:) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
        Denominator3(ci,:,:) = randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1);
    end
end

Denominator1 = sum(Denominator1, 1)*w + M;
Denominator2 = sum(Denominator2, 1)*w + M;
Denominator3 = sum(Denominator3, 1)*w + M;
% D = [D1; D2; D3];
% The product of divisive normalization before adding late noise
DNP = K*Numerator./[Denominator1; Denominator2; Denominator3];
% clear D;
clear Denominator1 Denominator2 Denominator3;
if gpuparallel
    SVs = DNP + gpuArray.randn(size(Numerator))*eta;
else
    SVs = DNP + randn(size(Numerator))*eta;
end
clear DNP;
CVs = squeeze(std(SVs, [], 2)./mean(SVs, 2));
% SVs = SVs(1:2,:,:);
max_from_each_distribution = SVs == max(SVs, [], 1);
probs = squeeze(sum(max_from_each_distribution, 2) / size(SVs, 2));
clear max_from_each_distribution;

Ovlps = nan([1, numel(dat.V3)]);
if gpuparallel
    SVs = gather(SVs);
end
dSVrng = [min(SVs(:)), max(SVs(:))];
for v3i = 1:numel(dat.V3)
    pd1 = fitdist(SVs(1,:,v3i)','kernel','Kernel','normal');
    x = dSVrng(1):.1:dSVrng(2);
    y1 = pdf(pd1, x);
    pd2 = fitdist(SVs(2,:,v3i)','kernel','Kernel','normal');
    y2 = pdf(pd2, x);
    ovlp = sum([y2(y1>y2), y1(y2>y1)])*.1;
    Ovlps(v3i) = ovlp*100;
end
if gpuparallel
    probs = gather(probs);
    CVs = gather(CVs);
end
end