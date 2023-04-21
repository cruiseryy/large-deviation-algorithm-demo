import numpy as np

class mc_sampler:
    def __init__(self, thres=1, data=np.zeros([39,90])) -> None:
        self.thres = thres
        data[np.where(data<=self.thres)] = 0
        self.data = data[np.argsort(np.sum(data, axis=1)),:]
        self.T, self.N = self.data.shape
        pause = 1
        return

    def get_para(self):
        pp = [[[0]*2 for _ in range(2)] for _ in range(2)]
        for i in range(self.T):
            for j in range(3, self.N):
                f1 = 0 if self.data[i, j-2] <= self.thres else 1
                f2 = 0 if self.data[i, j-1] <= self.thres else 1
                f3 = 0 if self.data[i, j] <= self.thres else 1
                pp[f1][f2][f3] += 1
        ppp = [[0]*2 for _ in range(2)]
        for i in range(2):
            for j in range(2):
                ppp[i][j] = pp[i][j][0] + pp[i][j][1]
                pp[i][j][0], pp[i][j][1] = pp[i][j][0]/ppp[i][j], pp[i][j][1]/ppp[i][j]
        ppp[0][0], ppp[0][1] = ppp[0][0]/(ppp[0][0]+ppp[0][1]), ppp[0][1]/(ppp[0][0]+ppp[0][1])
        ppp[1][0], ppp[1][1] = ppp[1][0]/(ppp[1][0]+ppp[1][1]), ppp[1][1]/(ppp[1][0]+ppp[1][1])
        self.pp = pp 
        self.ppp = ppp
        self.prcp = self.data[np.where(self.data > self.thres)]
        self.p1 = len(self.prcp) / self.T / self.N
        self.p0 = 1 - self.p1
        pause = 1
        return 
    
    def generate(self, dt=90, ic=[]):
        traj = np.zeros([dt+2,])
        if len(ic) == 0:
            d = np.random.rand()
            if d <= self.p1:
                traj[0] = np.random.choice(self.prcp)
            f1 = 0 if traj[0] == 0 else 1 
            d = np.random.rand()
            if d <= self.ppp[f1][1]:
                traj[1] = np.random.choice(self.prcp)
        elif len(ic) == 1:
            traj[0] = ic[0]
            f1 = 0 if traj[0] == 0 else 1 
            d = np.random.rand()
            if d <= self.ppp[f1][1]:
                traj[1] = np.random.choice(self.prcp)
        else:
            traj[:2] = ic
        for i in range(2, len(traj)):

            f1 = 0 if traj[i-2] == 0 else 1
            f2 = 0 if traj[i-1] == 0 else 1

            d = np.random.rand()
            if d < self.pp[f1][f2][1]:
                traj[i] = np.random.choice(self.prcp)
        
        return traj[2:]

class mc_sampler2:
    def __init__(self, thres=1, data=np.zeros([39,90])) -> None:
        self.thres = thres
        data[np.where(data<=self.thres)] = 0
        self.data = data[np.argsort(np.sum(data, axis=1)),:]
        self.T, self.N = self.data.shape
        pause = 1
        return

    def get_para(self):
        pp = [[0]*2 for _ in range(2)]
        for i in range(self.T):
            for j in range(2, self.N):
                f1 = 0 if self.data[i, j-1] <= self.thres else 1
                f2 = 0 if self.data[i, j] <= self.thres else 1
                pp[f1][f2] += 1
        p0 = pp[0][0] + pp[0][1]
        p1 = pp[1][0] + pp[1][1]
        pp[0][0], pp[0][1] = pp[0][0]/p0, pp[0][1]/p0
        pp[1][0], pp[1][1] = pp[1][0]/p1, pp[1][1]/p1

        self.prcp = self.data[np.where(self.data > self.thres)]

        self.pp = pp 
        self.p0 = p0
        self.p1 = p1
        return 
    
    def generate(self, dt=90, ic=[]):
        traj = np.zeros([dt+1,])
        if len(ic) == 0:
            d = np.random.rand()
            if d <= self.p1:
                traj[0] = np.random.choice(self.prcp)
        else:
            traj[0] = ic[0]
        for i in range(1, len(traj)):
            f1 = 0 if traj[i-1] == 0 else 1
            
            d = np.random.rand()
            if d <= self.pp[f1][1]:
                traj[i] = np.random.choice(self.prcp)

        return traj[1:]

        

if __name__ == "__main__":
    test = mc_sampler()
    test.modify(flag=0)
    test.get_para()

        
    