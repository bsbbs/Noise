function [probs, Ovlps, CVs] = dDNbFig1(pars, dat, num_samples) 
%% % Bo Shen, NYU, 2024
% distributional divisive normalization
% type b: cut inputs as non-negative, assume inputs noise as fully independent
if gpuDeviceCount > 0
    gpuparallel = 1;
else
    gpuparallel = 0;
end

eta = pars(1); % late noise standard deviation
scl = pars(2); % scaling for early noise
wp = pars(3); % weight of normalization
Mp = pars(4); % baseline normalization
Rmax = 75;
data = dat(:, {'V1', 'V2', 'V3', 'sdV1','sdV2','sdV3'});
Ntrl = size(dat,1);
samples = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
        stds = gpuArray(data.(['sdV', num2str(ci)])')*scl;
        samples(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
    else
        values = data.(['V',num2str(ci)])';
        stds = data.(['sdV', num2str(ci)])'*scl;
        samples(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
    end
end
D1 = [];
D2 = [];
D3 = [];
for ci = 1:3
    if gpuparallel
        values = gpuArray(data.(['V',num2str(ci)])');
        stds = gpuArray(data.(['sdV', num2str(ci)])')*scl;
        D1(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        D2(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        D3(ci,:,:) = max(gpuArray.randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
    else
        values = data.(['V',num2str(ci)])';
        stds = data.(['sdV', num2str(ci)])'*scl;
        D1(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        D2(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
        D3(ci,:,:) = max(randn([num_samples, Ntrl]).*stds + repmat(values, num_samples, 1), 0);
    end
end

D1 = sum(D1, 1)*wp + Mp;
D2 = sum(D2, 1)*wp + Mp;
D3 = sum(D3, 1)*wp + Mp;
D = [D1; D2; D3];
clear D1 D2 D3;
% The product of divisive normalization before adding late noise
DNP = Rmax*samples./D;
if gpuparallel
    SVs = max(DNP + gpuArray.randn(size(samples))*eta, 0);
else
    SVs = max(DNP + randn(size(samples))*eta, 0);
end
max_from_each_distribution = SVs == max(SVs, [], 1);
probs = squeeze(sum(max_from_each_distribution, 2) / size(SVs, 2));
CVs = squeeze(std(SVs, [], 2)./mean(SVs, 2));

Ovlps = nan([1, numel(dat.V3)]);
dSVrng = [min(SVs(:)), max(SVs(:))];
if gpuparallel
    SVs = gather(SVs);
end
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