%% 导入数据
clear,clc;

%% motion
% load motion.mat
% data={X_person1,X_person2};
% truelabel{1}=Y_person1';
% C=max(truelabel{1});
% [~,H]=size(data);
% for h=1:H
%     data{h}=mapminmax(data{h},0,1);
%     data{h}=data{h}';
% end

%% 初始化
[n,~]=size(data{1});


times=10;
AC1=zeros(times,1);
nmi=zeros(times,1);
RI=zeros(times,1);
metrics=[];
stdmetrics=[];
%% 开始计算
for time=1:times
    loop=1;
    lambda=zeros(1,H);
    U=zeros(n,C);
    W=ones(C,H);
    center = cell(1,H);
    target=0;
    for h=1:H
        lambda(h)=getBeta(data{h});
        col=size(data{h},2);
        center{h}=zeros(C,col);
    end
    %% 初始化聚类中心
    for h=1:H
%                 center{h}=fcm(data{h},C);
%                          [~,center{h}]=kmeans(data{h},C);
        %center{h}=Centroids_Initialization(data{h},C);
        center{h}=get_random_center(data{h},C);
    end
    %% 开始迭代
    while true
        loop=loop+1;
        distence = zeros(n,C,H);
        new_center = cell(1,H);
        %% 计算距离
        for h =1:H
            for i = 1 : n
                for j = 1 : C
                    distence(i,j,h)=alternative_metric(data{h}(i,:),center{h}(j,:),lambda(h));
                end
            end
        end
        %% 更新隶属度
        for i=1:n
            dis=zeros(1,C);
            for j=1:C
                U(i,j)=0;
                for h=1:H
                    dis(j)=dis(j)+distence(i,j,h)*W(j,h);
                end
            end
            [~,p]=min(dis);
            U(i,p)=1;
        end
        %% 更新类中心
        for j=1:C
            for h = 1 : H
                col=size(data{h}(i,:),2);
                fz=zeros(1,col);
                fm=0;
                for i =1 : n
                    fz=fz+U(i,j)*exp(-lambda(h)*norm(data{h}(i,:)-center{h}(j,:))^2)*data{h}(i,:);
                    fm=fm+U(i,j)*exp(-lambda(h)*norm(data{h}(i,:)-center{h}(j,:))^2);
                end
                if(fm==0)
                    fm=1e-12;
                end
                new_center{h}(j,:)=fz/fm;
            end
        end
        center = new_center;
        %% 更新视图权重
        for j = 1:C
            for h=1:H
                W(j,h)=1;
                for k =1:H
                    fz=sum(U(:,j).*distence(:,j,h));
                    fm=sum(U(:,j).*distence(:,j,k));
                    if(fz==0)
                        fz=1e-12;
                    end
                    if(fm==0)
                        fm=1e-12;
                    end
                    W(j,h)=W(j,h)*(fz/fm);% 分母有可能为0
                end
                W(j,h)=1/W(j,h)^(1/H);
            end
        end
        %% 判断退出
        new_target=0;
        for h =1 :H
            for j=1:C
                new_target=new_target+sum(W(j,h)*U(:,j).*distence(:,j,h));
            end
        end
        if(abs(new_target-target(loop-1))<1e-5)
            break;
        end
        target(loop)=new_target;
    end
    %% 标签匹配，计算正确率
    label=zeros(n,1);
    for i=1:n
        [~,p]=max(U(i,:));
        label(i)=p;
    end
    result_label=label_map(label,truelabel{1});
    Y=truelabel{1};



    AC1(time) = length(find(Y == result_label))/length(Y);
    result = CalcMeasures(Y, result_label);
    nmi(time) = NMI(Y,result_label);
    [~,~,~,RI(time)] = Evaluate(Y,result_label);
end
meanAC=mean(AC1);
meanNMI=mean(nmi);
meanRI=mean(RI);
stdAC=std(AC1);
stdNMI= std(nmi);
stdRI= std(RI);
metrics = [metrics;[meanAC,meanNMI,meanRI]];
fprintf("ACC:%f,NMI:%f,RI:%f\n",meanAC,meanNMI,,meanRI);
stdmetrics = [stdmetrics;[stdAC,stdNMI,stdRI]];


