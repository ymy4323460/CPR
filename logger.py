import os

class logger:
    def __init__(self, data_dir, name, dataset):
        save_dir = os.path.join('logs')
        if not os.path.exists(save_dir):
           os.makedirs(save_dir)
        file_path = os.path.join(save_dir, '{}_{}.txt'.format(name, dataset))
        self.fid = open(file_path, 'a')

    def logging(self, info):
        self.fid.write(info)

    def shutdown(self, flag=False):
        if not flag:
            self.fid.write('---'*30+'\n')
            self.fid.close()
        else:
            # self.fid.write('***'*30+'\n\n')
            self.fid.write('\n')
            self.fid.close()
