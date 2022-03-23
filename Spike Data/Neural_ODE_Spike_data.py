import argparse
import matplotlib
matplotlib.use('agg')
import torch.optim as optim
from model import *
from Get_Data import *

from torch.optim.lr_scheduler import StepLR
'''
Tasks:
 - why has the decoder output the shape [natch size, time points, latent dim]
 - exact understanding of the inbuild loss
 - using several Units
 - using batches
 - scheiß laden von daten reparieren
 - improve saving directory detection
 - visualize backwards learning
 - find out how to avoid rk overflow t + dt > nan

'''
parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=True)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=15)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default='Storage/models')
parser.add_argument('--method', type=str, default='dopri5')
parser.add_argument('--data_type', type=str, default='not_Test_data')
parser.add_argument('--name_of_run', type=str, default='KL_Loss_adjoint')
parser.add_argument('--load_data', type=bool, default=True)
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--latent_dim', type=int, default=16)
parser.add_argument('--nhidden', type=int, default=80)
parser.add_argument('--rnn_nhidden', type=int, default=100)
parser.add_argument('--number_units', type=int, default=1)
parser.add_argument('--nbatch', type=int, default=1)
parser.add_argument('--data_resolution', type=int, default=1000)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

noise_std = .3
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

# get data
Trajectories, time_series = get_data(name=args.name_of_run,
                                                    load=args.load_data,
                                                    mode=args.data_type,
                                                    resolution_LFR=args.data_resolution,
                                                    obs_dim=args.number_units)

time_points = len(time_series)
section = np.max(np.abs([np.min(Trajectories), np.max (Trajectories)]))
Trajectories = torch.from_numpy(Trajectories).float().to(device)
time_series = torch.from_numpy(time_series).float().to(device)

# model
func = LatentODEfunc(args.latent_dim, args.nhidden).to(device)
rec = RecognitionRNN(latent_dim=args.latent_dim, obs_dim=args.number_units, nhidden=args.rnn_nhidden, nbatch=args.nbatch).to(device)
dec = Decoder(time_points=time_points, section=section, latent_dim=args.latent_dim, obs_dim=args.number_units, nhidden=args.nhidden).to(device)
params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
optimizer = optim.Adam(params, lr=args.lr)
scheduler = StepLR(optimizer, int(args.niters/4), gamma=0.5, verbose=True)
loss_meter = RunningAverageMeter()
loss_function = nn.MSELoss()

if args.load_model:
    if args.train_dir is not None:
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        ckpt_path = os.path.join(args.train_dir, args.name_of_run + 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            rec.load_state_dict(checkpoint['rec_state_dict'])
            dec.load_state_dict(checkpoint['dec_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            Trajectories = checkpoint['Trajectories']
            orig_ts = checkpoint['orig_ts']
            print('Loaded ckpt from {}'.format(ckpt_path))
else:
    func.apply(weights_init_normal)
    rec.apply(weights_init_normal)
    dec.apply(weights_init_normal)#

proceed = True
proceed_round=0

while proceed:
    LOSS = []
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        # backward in time to infer q(z_0)
        h = rec.initHidden().to(device)
        for t in reversed(range(Trajectories.size(1))):
            obs = Trajectories[:, t, :]
            out, h = rec.forward(obs, h)
        qz0_mean, qz0_logvar = out[:, :args.latent_dim], out[:, args.latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        # forward in time and solve ode for reconstructions

        pred_z = odeint(func, z0, time_series, method=args.method).permute(1, 0, 2)
        pred_x = dec(pred_z)

        ## compute loss
        noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
        noise_logvar = 2. * torch.log(noise_std_).to(device)
        logpx = log_normal_pdf(Trajectories, pred_x, noise_logvar).sum(-1).sum(-1)
        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar, pz0_mean, pz0_logvar).sum(-1)
        loss = torch.mean(-logpx + analytic_kl, dim=0)

        #loss = loss_function(Trajectories, pred_x)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        scheduler.step()
        LOSS.append(loss)

        print('Iter: {}, running avg elbo: {:.4f}'.format(itr + proceed_round, loss_meter.avg))
        #print('Iter: {}, running avg elbo: {:.4f}'.format(itr + proceed_round, loss))


    if args.train_dir is not None:
        ckpt_path = os.path.join(args.train_dir, args.name_of_run + 'ckpt.pth')
        torch.save({
            'func_state_dict': func.state_dict(),
            'rec_state_dict': rec.state_dict(),
            'dec_state_dict': dec.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'Trajectories': Trajectories,
            'time_series': time_series,
        }, ckpt_path)
        print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))

    if args.visualize:
        with torch.no_grad():
            # sample from trajectorys' approx. posterior
            h = rec.initHidden().to(device)
            for t in reversed(range(Trajectories.size(1))):
                obs = Trajectories[:, t, :]
                out, h = rec.forward(obs, h)
            qz0_mean, qz0_logvar = out[:, :args.latent_dim], out[:, args.latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            # take first trajectory for visualization
            z0 = z0[0]

            ts_pos = np.linspace(0., 2. * np.pi, num=time_points)
            ts_neg = np.linspace(-2. * np.pi, 0., num=time_points)[::-1].copy()
            ts_pos = torch.from_numpy(ts_pos).float().to(device)
            ts_neg = torch.from_numpy(ts_neg).float().to(device)

            zs_pos = odeint(func, z0, ts_pos, method=args.method)
            #zs_neg = odeint(func, z0, ts_neg, method=args.method)

            zs_pos = zs_pos.reshape(1, time_points, 16)
            #zs_neg = zs_neg.reshape(1, 100, 16)

            xs_pos = dec(zs_pos)
            #xs_neg = torch.flip(dec(zs_neg), dims=[0])

        xs_pos = xs_pos.cpu().numpy()[0, :, 0]
        #xs_neg = xs_neg.cpu().numpy()
        orig_traj = Trajectories[0].cpu().numpy()[:, 0]

        plt.figure()
        plt.plot(np.linspace(0, 1, len(orig_traj)), orig_traj, 'g', label='true trajectory')
        plt.plot(np.linspace(0, 1, len(xs_pos)), xs_pos, 'r', label='learned trajectory (t>0)')
        #plt.plot(np.linspace(0, 100, len(xs_neg[:, 0])), xs_neg[0, :, 0], 'c', label='learned trajectory (t<0)')
        mini = np.min(orig_traj)
        maxi = np.max(orig_traj)
        plt.ylim(mini - abs(mini) * 0.1, maxi + 0.1 * abs(maxi))
        plt.legend()
        plt.savefig('Storage/Result/Fit_' + args.name_of_run + '_{}'.format(args.data_resolution) + '.png', dpi=200)
        plt.figure()
        plt.plot(LOSS)
        plt.yscale('log')
        plt.savefig('Storage/Loss_curves/Loss_' + args.name_of_run + '_{}'.format(args.data_resolution) + '.png', dpi=200)
        print('Saved visualization figure at {}'.format('./vis.png'))

        # get input if continue training
        Question = input('Do you want to continue training? y/n')
        if Question == ("n"):
            exit()
        else:
            proceed_round += 1
