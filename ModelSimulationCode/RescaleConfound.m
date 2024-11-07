% Test the idea of scaled V3 will co-change with task difficulty after
% re-scaling
N = 55;
V1 = nan(15, N);
V2 = nan(15, N);
V3 = nan(6, N);
for subi = 1:N
    Vs = rand(12, 1)*randi(90);
    Vs = sort(Vs, 'descend');
    Trgts = nchoosek(Vs(1:6), 2);
    [Trs, V3s] = meshgrid(Trgts, Vs(7:end));
    V1(:, subi) = Trgts(:,1);
    V2(:, subi) = Trgts(:,2);
    V3(:, subi) = Vs(7:end);
end

%%
plot(V3(:), V1(:)-V2(:), '.');