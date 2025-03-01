import numpy as np
from numpy import pi,sin,cos,exp,abs,inf,nan
from wavedata import Wave,Wave2D,wrange,profile,excelcolumn,csvcolumns,deal,loadvna,dotdict,widths2grid,timeit,track
from plot import plot,multiplot,multiplots
from sellmeier import index,pidbm
import modes
import joblib
np.seterr(divide='ignore')
memory = joblib.Memory('j:/backup', verbose=0) # use as @memory.cache
relativepermittivity = {'ktp':(13,13),'xcutln':(28,43),'xcutmgln':(28,43),'ln':(43,28),'mgln':(43,28),'tfln':(28,43),'isoln':34.7,
    'xcutlndc':(29,85),'lndc':(85,29),'mglndc':(85,29),'tflndc':(29,85),'isolndc':49.6,
    'air':1,'corning7070':4.1,'newsub':4.6,'buffer':4,'epoxy':4,'silica':3.9,'atp silica':3.826,'sio2':3.9,'sio₂':3.9,'quartz':4,' ':4,'':4,'sub':4,
    'silicon':11.7,'aluminum nitride':8.6,'alumina':9,'fr4':4.35,'bcb':2.57} # relativepermittivity = (εy,εz) = (horiz,vert), dc = unclamped ε values
conductivity = {'aluminum':38.16,'chromium':38.46,'copper':58.13,'gold':40.98,'nickel':14.49,'silver':61.73,'tungsten':18.25,} # 1/(Ω·µm) # from Pozar Appendix G
losstangent = {'alumina':0.0003,'fusedquartz':0.0001,'pyrex':0.0054,'silicon':0.0040,'teflon':0.0004,'silicondioxide':0.0008,'corning7070':0.0006} # from Pozar Appendix H # SiO2 from https://arxiv.org/pdf/2304.01362.pdf # https://www.rfcafe.com/references/electrical/dielectric-constants-strengths.htm
# cachefolder = 'd:/cache/elmer/'
cachefolder = 'c:/temp/elemercache'
elmerverbose = 0#True
skindepthwarn = True
# from rich.progress import track

@memory.cache
def overlapintegral(iizz,iixs,iiys,eezz,eexs,eeys,gap,xoffset=0,method='linear',deadregion=None): # deadregion = (Δwidth,Δdepth) of dead region
    ii,ee = Wave2D(iizz,xs=iixs,ys=iiys),Wave2D(eezz,xs=eexs,ys=eeys)
    # Γ = s/V0 ∫∫Em|E²|dxdy / ∫∫|E²|dxdy
    # ref: Fuste13 - Design Rules and Optimization of Electro-Optic Modulators Based on Coplanar Waveguides.pdf, page 12
    def xys():
        for i,j in np.ndindex(ii.shape):
            yield ii.xx[i,j],ii.yy[i,j]
    def isnotdead(x,y):
        return True if deadregion is None else not (-deadregion[0]/2<x<deadregion[0]/2 and -abs(deadregion[1])<y<0)
    conv = sum([ ii(x,y) * ee(x+xoffset,y,method=method) * isnotdead(x,y) for x,y in xys() ])
    norm = sum([ ii(x,y) for x,y in xys() ])
    return conv/norm * gap/1.
# α = 1/e falloff distance for voltage, 1/e² for power
def loss2α(loss): # loss in dB/cm, α in cm⁻¹
    return np.log(10)*loss/20
def α2loss(α): # α in cm⁻¹, loss in dB/cm
    return 20*α/np.log(10)
def vpp2dbm(v,Z=50):
    return 10*np.log10(125*v**2/Z)
def dbm2vpp(dbm,Z=50):
    return np.sqrt(10**(dbm/10)*Z/125)
def pidbm(vpi,Z=50): # power in dBm for peak-to-peak π phase shift
    return 10*np.log10(125*vpi**2/Z)
def pmintensity(n,beta): # sideband intensity given n = sideband number, beta = pi*V/Vpi
    from scipy.special import jv
    return jv(n,beta)**2
def firstsidebandratio(β): # β = πV/Vπ = ½πVpp/Vπ
    return pmintensity(1,β)/pmintensity(0,β)
def βfromfirstsidebandratio(r): # β = πV/Vπ = ½πVpp/Vπ
    from scipy.optimize import brentq
    return brentq(lambda β: firstsidebandratio(β)-r,0,1)
def firstsidebandratio2vpi(dbm,r,Z=50): # r = firstsidebandratio = intensity ratio of first sideband to carrier
    β = βfromfirstsidebandratio(r) # β = πV/Vπ = ½πVpp/Vπ
    vpp = dbm2vpp(dbm,Z=Z)
    vpi = 0.5*pi*vpp/β
    return vpi
def rffreq2wavelength(f,nrf): # f in GHz, nrf = refractive index, 
    # returns λ in mm
    return 299792458/(f*1e9*nrf)*1e3

# @memory.cache
# def modesolver(*args,**kwargs):
#     return modes.modesolver(*args,**kwargs)
class Electrode():
    def __init__(self, material, layers, gridx, gridy, gridnum, stretch=100, margin=10, modeargs=None):  # layers are listed bottom to top (sublisted left to right)
        # print( '\nmaterial',material, '\nlayers',layers, '\ngridx',gridx, '\ngridy',gridy, '\ngridnum',gridnum)
        # self.args = {k:v for k,v in locals().items() if not 'self'==k}
        self.material,self.layers,self.gridx,self.gridy,self.gridnum,self.stretch,self.margin,self.modeargs = material,layers,gridx,gridy,gridnum,stretch,margin,modeargs
        self.xstretch,self.ystretch = None,None
        assert 'hot' in self.id() and 'ground' in self.id(), f'hot and ground must be defined: {material}'
        assert max([v for k,v in self.id().items() if k not in ['hot','ground']])<self.id()['hot']<self.id()['ground'], f'ground must have highest id, hot next highest: {material}'
        self.results_ = None
        self.loss_ = None
        self.md_,self.ii_ = None,None
    def legendtext(self,xoffset=None,loss=True):
        s = f'Z={self.Z:.1f}Ω\nC={self.C:.2f}pF/cm\nL={self.L:.2f}nH/cm\n$n_{{RF}}$={self.nrf:.2f}'
        if self.modeargs is None:
            return s
        s += (f', $n_{{IR}}$={self.nir():.2f}\nVπ={self.vpis()(xoffset):.1f}V·cm at x={xoffset:.1f}µm' if xoffset is not None else 
              f', $n_{{IR}}$={self.nir():.2f}\nVπ={self.vpi():.1f}V·cm at x={self.bestoffset():.1f}µm')
        return s + f'\nloss={self.rfloss():.2f}dB/cm/√GHz' if loss else s
    def savetext(self): return f'electrode' # 'gap-hot-gap, {hot}µm hot, {metal}µm metal, 10µm LN on {substrate}'
    def dielectrics(self):
        return {k:v for k,v in self.material.items() if self.material[k] not in ['hot','ground']}
    def id(self):
        return {v:k for k,v in self.material.items()}
    def materialtext(self,margin=None):
        # return [{'x':self.gridx[1]-self.margin/2,'y':min(self.gridy[i+1],self.gridy[-2]+self.margin)-self.margin/2,'s':self.material[ks[0]]} for i,ks in enumerate(self.filledoutlayers()) if 'hot' not in [self.material[ks[j]] for j in range(len(ks))]] # self.material[ks[len(ks)//2]] not in ['hot','ground']]
        gx,gy,mm = np.array(self.gridx), np.array(self.gridy), 0.5*(margin if margin is not None else self.margin)
        gx,gy = [np.clip(x,gx[1:-1].min()-mm,gx[1:-1].max()+mm) for x in gx], [np.clip(y,gy[1:-1].min()-mm,gy[1:-1].max()+mm) for y in gy]
        return [{'x':gx[1]-mm, 'y':0.5*(gy[i]+gy[i+1]), 's':self.material[ks[0]], 'verticalalignment':'center'} 
            for i,ks in enumerate(self.filledoutlayers()) if 'hot' not in [self.material[ks[j]] for j in range(len(ks))]]
    def filledoutlayers(self): # e.g. input: [1,2,[1,3,1]], output: [[1 1 1][2 2 2][1 3 1]]
        rowlength = max([len(a) for a in self.layers if hasattr(a,'__len__')])
        return np.array([(a if hasattr(a,'__len__') else [a for _ in range(rowlength)]) for a in self.layers]).astype(int)
    def tilesize(self,i,j):
        x,y = self.gridx,self.gridy
        return (x[i+1]-x[i]),(y[j+1]-y[j])
    def shrinkindices(self,f,metal):
        metalindices = self.materialindices('hot')+self.materialindices('ground')
        metalmindim = min([x for i,j in metalindices for x in self.tilesize(i,j)])
        ii,jj = sorted(list(set([i for i,j in metalindices]))),sorted(list(set([j for i,j in metalindices])))
        d = 0.5*skindepth(f,metal=metal)
        if skindepthwarn:
            if metalmindim<4*d: print(f'warning, skindepth {2*d:g}µm, metal {metalmindim:g}µm at {f}GHz')
            # assert 2*d<metalmindim, f'skindepth too large ({2*d:g}µm) for metal ({metalmindim:g}µm), choose larger freq'
        return ii,jj,min(d,0.5*metalmindim-1e-6)
    def shrinkmetal(self,f,metal):
        ii,jj,d = self.shrinkindices(f,metal=metal)
        def isadjacent(ii):
            return any(i0+1==i1 for i0,i1 in zip(ii[:-1],ii[1:]))
        assert not isadjacent(ii) and not isadjacent(jj), 'adjacent metal tiles not yet supported'
        def shrinkgrid(grid,ii): # grid = gridx or gridy, ii = indices to shrink in size
            return [((g if 0==i else g+d) if i in ii else (g if len(grid)-1==i else g-d) if i-1 in ii else g) for i,g in enumerate(grid)]
        gridx = shrinkgrid(self.gridx,ii)
        gridy = shrinkgrid(self.gridy,jj)
        return gridx, gridy
    def skinloss(self,f,metal='gold',verbose=False,db=False,ishrink=None,jshrink=None): # loss calc using Wheeler incremental inductance rule, Pozar p.76
        if verbose: print(f'shrinking metal by {0.5*skindepth(f,metal=metal)}µm on all sides')
        L0,Z0 = self.L,self.Z #; print(f"L₀={L0:g}nH/cm, Z₀={Z0:g}Ω")
        gridx, gridy = self.shrinkmetal(f,metal)
        el = Electrode(self.material,self.layers, gridx, gridy, gridnum=self.gridnum,stretch=self.stretch,margin=self.margin,modeargs=self.modeargs)
        L = el.L #; print(f"L={L:g}nH/cm")
        α = pi*f*(L-L0)/Z0
        loss = 20*α/np.log(10)/np.sqrt(f)
        if verbose: print('αac',α,'cm⁻¹, loss',loss,'dB/cm/√GHz')
        return loss if db else α
    def rfloss(self,metal='gold',f0=10,f1=40,verbose=False): # dB/cm/√GHz
        if self.loss_ is not None:
            return self.loss_
        α0,α1 = [self.skinloss(f=f,metal=metal,verbose=verbose,db=False) for f in (f0,f1)]
        self.loss_ = (α2loss(α1)-α2loss(α0))/(np.sqrt(f1)-np.sqrt(f0))
        return self.loss_
    def dcresistance(self): # in Ω/cm
        rg = 1e4/conductivity['gold']/self.materialarea('ground')
        rh = 1e4/conductivity['gold']/self.materialarea('hot')
        # print('\n',' rg',rg,'Ω/cm ground','\n',' rh',rh,'Ω/cm hot','\n',' r',rh+rg,'Ω/cm total')
        return rg+rh
    def resistance(self,f): # in Ω/cm
        return self.dcresistance() + 2 * self.Z * loss2α(np.sqrt(f)*self.rfloss())
    def materialindices(self,s='hot'):
        def layersexpanded(self):
            return [(row if hasattr(row,'__len__') else (len(self.gridx)-1)*[row]) for row in self.layers]
        l = layersexpanded(self)
        l = self.filledoutlayers()
        id = [k for k in self.material if self.material[k]==s][0]
        nx,ny = len(self.gridx)-1,len(self.gridy)-1
        return [(i,j) for j in range(ny) for i in range(nx) if id==l[j][i]]
    def materialarea(self,s='hot'):
        def gridarea(self,i,j):
            x,y = self.gridx,self.gridy
            return (y[j+1]-y[j]) * (x[i+1]-x[i])
        indices = self.materialindices(s=s)
        return sum([gridarea(self,i,j) for i,j in indices])
    def scanres(self,ress=None):
        def r(gn):
            def npfloat(a): return np.array(a).astype(float)
            def asfloat(x): return x if x is None else float(x)
            return electricsolve(self.material,self.filledoutlayers()[::-1],npfloat(self.gridx),npfloat(self.gridy),int(gn),asfloat(self.stretch),asfloat(self.xstretch),asfloat(self.ystretch))
        wx = ress if ress is not None else 36000 * 2**np.linspace(1,8,8)
        wy = [r(gn) for gn in wx]
        wz,wn = [y['Z'] for y in wy],[y['nrf'] for y in wy]
        rx,ry = [np.diff(y['xx']).min() for y in wy],[np.diff(y['yy']).min() for y in wy]
        print(f' Z{wz}\n nrf{wz}')
        Wave.plots(Wave(wz,wx),logx=1,m=1,x='grid points',y='Z (Ω)',save='z res test')
        Wave.plots(Wave(wn,wx),logx=1,m=1,x='grid points',y='$n_{RF}$',save='nrf res test')
        Wave.plots(Wave(rx,wx),logx=1,m=1,x='grid points',y='Δx (µm)',save='Δx res test')
        Wave.plots(Wave(ry,wx),logx=1,m=1,x='grid points',y='Δy (µm)',save='Δy res test')
    def hx(self): # return Wave2D of magnetic field Hx # H = √(ε/µ₀) z x E
        ...
    def hy(self): # return Wave2D of magnetic field Hy # H = √(ε/µ₀) z x E
        ...
    def results(self):
        if self.results_ is None:
            def npfloat(a): return np.array(a).astype(float)
            def asfloat(x): return x if x is None else float(x)
            gridsubs = self.filledoutlayers()[::-1] # layers are bottom to top, gridsubs are top to bottom
            gridx,gridy = npfloat(self.gridx),npfloat(self.gridy)
            self.results_ = electricsolve(self.material,gridsubs,gridx,gridy,int(self.gridnum),asfloat(self.stretch),asfloat(self.xstretch),asfloat(self.ystretch))
        return self.results_
    def __getattr__(self,name):
        name = 'v' if 'potential'==name else name
        if name in ['C','L','Z','Z0','nrf','v','ex','ey','dvdx','dvdy','xx','yy']:
            return self.results()[name]
        if name in ['λ','sell']:
            return self.modeargs[name]
        assert 0, f'{name} is not a property of Electrode'
    def minres(self): return np.diff(self.xx).min(),np.diff(self.yy).min()
    def md(self):
        if self.md_ is None:
            self.md_ = self.modeargs['md'] if 'md' in self.modeargs else modes.modesolver(**self.modeargs)
        return self.md_
    def ii(self):
        if self.ii_ is None:
            ii = self.md().ee**2
            self.ii_ = ii[:,:int(ii.y2p(-1e-3))+1] # exclude y>=0  ## also limit ii to x and y range where ii is greater than some cutoff (for speed)?
        return self.ii_
    def nir(self):
        return self.md().neff
    def bestlength(self,f): # in mm # electrode length for best Vpi (no rf loss)
        return 1000*299792458/(2*1e9*f*np.abs(self.nrf-self.nir()))
    def efield(self):
        return self.dvdx if self.modeargs.get('xcut',False) else self.dvdy
    def overlapintegralgamma(self,gap=None,xoffset=0,method='linear',dead=False):
        gap = gap if gap is not None else self.gap
        iizz,iixs,iiys = self.ii().array(),self.ii().xs,self.ii().ys
        eezz,eexs,eeys = self.efield().array(),self.efield().xs,self.efield().ys
        return overlapintegral(iizz,iixs,iiys,eezz,eexs,eeys,gap,xoffset=xoffset,method='linear',
            deadregion=(self.md().args.width,self.md().args.depth) if dead else None)
        # return overlapintegral(self.ii(),self.efield(),gap,xoffset=xoffset,method='linear',
        #     deadregion=(self.md().args.width,self.md().args.depth) if dead else None) ### AttributeError: 'function' object has no attribute 'view' # line 205 in "C:\Python310\Lib\site-packages\joblib\hashing.py"
        # def overlapintegralgamma(self,gap,xoffset=0,method='linear',dead=False):
        #     # Γ = s/V0 ∫∫Em|E²|dxdy / ∫∫|E²|dxdy
        #     # ref: Fuste13 - Design Rules and Optimization of Electro-Optic Modulators Based on Coplanar Waveguides.pdf, page 12
        #     ii = self.ii()
        #     def xys():
        #         for i,j in np.ndindex(ii.shape):
        #             yield ii.xx[i,j],ii.yy[i,j]
        #     def isnotdead(x,y):
        #         w,d = self.md().args.width,self.md().args.depth # print('width,depth',width,depth)
        #         return not (dead and -w/2<x<w/2 and -d<y<0)
        #     conv = sum([ ii(x,y) * self.efield()(x+xoffset,y,method=method) * isnotdead(x,y) for x,y in xys() ])
        #     norm = sum([ ii(x,y) for x,y in xys() ])
        #     return conv/norm * gap/1.
    def estimatedvpi(self,gap,λ=None,sell=None,getdn=False):
        λ = λ if λ is not None else self.modeargs['λ']
        sell = sell if sell is not None else self.modeargs['sell']
        r33 = 32 if sell.startswith('tfln') or sell.startswith('ln') else {'ktp':36.3,'ln':32,'mgln':32,'mglnridge':32}[sell]
        n0 = index(λ,sell.replace('ridge',''),20)
        eodn = 0.5 * n0**3 * r33 * 1e-9 # delta n for 1 V/mm field
        phase = 2*pi*(eodn)*1e10/λ
        return gap*pi/phase if not getdn else eodn
    def phaseperfieldpercm(self): # phase shift for 1 V/µm field, 1 cm length
        λ,sell = self.modeargs['λ'],self.modeargs['sell']
        r33 = 32 if sell.startswith('tfln') or sell.startswith('ln') else {'ktp':36.3,'ln':32,'mgln':32,'mglnridge':32,'mglnridgez':32}[sell]
        sell = 'ln' if sell.startswith('tfln') or sell.startswith('ln') else sell.replace('ridge','')
        n0 = index(λ,sell,20) # n0 = self.md().neff # can't use self.md().nsub for mglnridge
        eodn = 0.5 * n0**3 * r33 * 1e-9 # delta n for 1 V/mm field
        phase = 2*np.pi*(eodn)*1e10/λ
        return phase
    def vpis(self,dead=False,xs=None):
        xs = np.linspace(-100,100,2001) if xs is None else xs
        fs = Wave(abs(self.overlapintegralgamma(1,xs,'cubic',dead=dead)),xs) # field seen by mode assuming 1 volt applied
        λ,sell = self.modeargs['λ'],self.modeargs['sell']
        r33 = 32 if sell.startswith('tfln') or sell.startswith('ln') else {'ktp':36.3,'ln':32,'mgln':32,'mglnridge':32,'mglnridgez':32}[sell]
        # eodn = 0.5 * (self.md().nsub)**3 * r33 * 1e-9  # delta n for 1 V/mm field
        # phase = 2*np.pi*(eodn)*1e10/λ  # phase shift for 1 V/µm field, 1 cm length
        # field, vpi, vpis = fs(fs.xmax()), np.pi/phase/fs(fs.xmax()), np.pi/phase/fs # print(f'phase for 1V/µm: {phase:.2f}rad,',f'max field for 1V: {field:.2f}V/µm,',f'position of max field: x = {fs.xmax():.1f}µm,',f'max phase: {phase*field:.2f}rad,',f'Vπ: {vpi:.2f}volt·cm')
        # print(np.pi/phase/fs,np.pi/self.phaseperfieldpercm()/fs)
        return np.pi/self.phaseperfieldpercm()/fs
    def bestoffset(self,dead=False,xs=None):
        return self.vpis(dead=dead,xs=xs).minloc()
    def vpi(self,xoffset=None,dead=False,db=0,length=10): # units of V·cm by default
        C = 10**(abs(db)/20) * 10/length
        # print(f"notdead:{self.vpis(dead=0).min():g}, dead:{self.vpis(dead=1).min():g}, vpi:{self.vpis(dead=dead).min():g}")
        return C*self.vpis(dead=dead).min() if xoffset is None else C*self.vpis(dead=dead,xs=np.array([xoffset])).atx(xoffset,monotonicx=False)
    def bandwidth(self,length,loss,za=50,zt=None,fmax=40,df=0.01,plot=0,dbm=False,xoffset=None,dead=False,verbose=True):
        def pidbm(vpi,Z=50): # power in dBm for peak-to-peak π phase shift
            return 10*np.log10(125*vpi**2/Z)
        zt = zt if zt is not None else self.Z
        self.d = rfbandwidth(Z=self.Z,nrf=self.nrf,loss=loss,za=za,zt=zt,lengthinmm=length,λ=self.λ,sell=self.sell+'wg',fmax=fmax,df=df,plot=plot)
        if dbm:
            ## power required for peak-to-peak phase of π
            ## Vpp = Vπ, Vrms = Vpp/(2√2) # P(W) = Vrms²/Z = ⅛Vpp²/Z
            ## P(dBm) = 10×log10(1000×P(W)) = 10×log10(125×Vpp²/Z)
            vπ = self.vpi(xoffset=xoffset,length=length,dead=dead)
            if verbose: print('vπ',vπ,'Z',self.Z,'nrf',self.nrf)
            assert pidbm(vπ,self.Z)==10*np.log10(125*vπ**2/self.Z)
            return pidbm(vπ,self.Z) - self.d['coprop']
        return self.d
    def plotlimits(self,xlim,ylim,aspect=None):
        gx,gy,mm = self.gridx,self.gridy,self.margin
        xlim,ylim = (gx[1]-mm,gx[-2]+mm) if xlim is None else xlim,(gy[1]-mm,gy[-2]+mm) if ylim is None else ylim
        if aspect:
            y = max(0, 0.6*(xlim[1]-xlim[0]) - (ylim[1]-ylim[0]))
            ylim = [ylim[0]-y/2,ylim[1]+y/2]
        return xlim,ylim
    def plotmode(self,xoffset=None,savetext='',xlim=None,ylim=None,materialtext=None,**kwargs):
        xoffset = xoffset if xoffset is not None else self.bestoffset()
        xlim,ylim = self.plotlimits(xlim,ylim,aspect='aspect' in kwargs)
        ii = self.md().ee**2
        ii.xs = ii.xs + xoffset
        legendtext = kwargs.pop('legendtext',self.legendtext(xoffset))
        plot(contourf=self.potential,contour=ii,xlabel='x (µm)',ylabel='y (µm)',xlim=xlim,ylim=ylim,
            lines=self.plotlines(),texts=materialtext if materialtext is not None else self.materialtext(),
            legendtext=legendtext,corner='upper right',
            save=(savetext if savetext else 'Optical mode and E-field'),**kwargs)
    def plotvpis(self,savetext='',dead=False,**kwargs):
        if 'log' in kwargs:
            self.vpis(dead=dead).plot(seed=1,xlim=(-30,30),ylim=(None,1e2),xlabel='x (µm)',ylabel='Vπ (volt·cm)',save='Vpi vs waveguide position'+(', '+savetext if savetext else ''),**kwargs)
        else:
            self.vpis(dead=dead).plot(seed=1,xlim=(-30,30),ylim=(0,3*self.vpi()),xlabel='x (µm)',ylabel='Vπ (volt·cm)',save='Vpi vs waveguide position'+(', '+savetext if savetext else ''),**kwargs)
    def plotgamma(self,gap=None,savetext='',dead=False,**kwargs):
        xs = np.linspace(-100,100,2001)
        gamma = Wave(self.overlapintegralgamma(gap,xs,'cubic',dead=dead),xs)
        gamma.plot(seed=2,xlabel='x (µm)',ylabel='overlap integral, Γ',save='overlap integral vs waveguide position'+(', '+savetext if savetext else ''),**kwargs)
    def plotlines(self):
        gs,gx,gy = self.filledoutlayers(),self.gridx,self.gridy
        def lines():
            for j,(y0,y1) in enumerate(zip(self.gridy[:-1],self.gridy[1:])):
                for i,x in enumerate(self.gridx[1:-1]): # print(i,j,x,(y0,y1),gs[j][i],gs[j][i+1])
                    if not gs[j][i]==gs[j][i+1]:
                        yield {'xdata':(x,x),'ydata':(y0,y1),'color':'k','linewidth':0.5} # yield plt.Line2D((x,x),(y0,y1),color='k')#,lw=2.5)
            for i,(x0,x1) in enumerate(zip(self.gridx[:-1],self.gridx[1:])):
                for j,y in enumerate(self.gridy[1:-1]):
                    if not gs[j][i]==gs[j+1][i]:
                        yield {'xdata':(x0,x1),'ydata':(y,y),'color':'k','linewidth':0.5} # yield plt.Line2D((x0,x1),(y,y),color='k')#,lw=2.5)
        return list(lines())
    def plot(self,texts=None,savetext='',xlim=None,ylim=None,**kwargs):
        # gx,gy,mm = self.gridx,self.gridy,self.margin
        # # xlim,ylim = (1.5*gx[1]-0.5*gx[2],1.5*gx[-2]-0.5*gx[-3]),(1.5*gy[1]-0.5*gy[2],1.5*gy[-2]-0.5*gy[-3])
        # xlim,ylim = (gx[1]-mm,gx[-2]+mm) if xlim is None else xlim,(gy[1]-mm,gy[-2]+mm) if ylim is None else ylim
        xlim,ylim = self.plotlimits(xlim,ylim,aspect='aspect' in kwargs)
        self.potential.plot(contourf=1,x='µm',y='µm',lines=self.plotlines(),
            texts=self.materialtext() if texts is None else texts,
            legendtext=self.legendtext(),corner='upper right',xlim=xlim,ylim=ylim,
            save='potential'+(', '+savetext if savetext else ', '+self.savetext()),**kwargs)
    def __str__(self):
        return f'C={self.C:.3f}pF/cm, L={self.L:.3f}nH/cm, Z={self.Z:.2f}Ω, Z0={self.Z0:.2f}Ω, nrf={self.nrf:.3f}'
class TileElectrode(Electrode):
    ''' for example:
        material = {1:KTP,2:'air',3:'hot',4:'ground'}
        tiles = [[2,2,2], # material for each tile, top to bottom
                 [3,2,4],
                 [1,1,1]]
        tilex = [500,10,500]   # width of each column in µm
        tiley = [500, 4,500]   # height of each row in µm
    '''
    def __init__(self, material, tiles, tilex, tiley, gridnum, modeargs=None, x0=None, y0=None):
        gridx,gridy = widths2grid(tilex,x0=x0),widths2grid(tiley,x0=y0)
        layers = tiles[::-1]
        super().__init__(material, layers, gridx=gridx, gridy=gridy, gridnum=gridnum, modeargs=modeargs)
class ShieldElectrode(TileElectrode):
    def __init__(self, material, tiles, tilex, tiley, shrinkii, shrinkjj, gridnum, modeargs=None, x0=None, y0=None):
        self.shrinkii,self.shrinkjj = shrinkii,shrinkjj
        super().__init__(material=material, tiles=tiles, tilex=tilex, tiley=tiley, gridnum=gridnum, modeargs=modeargs, x0=x0, y0=y0)
    def shrinkindices(self,f,metal):
        return self.shrinkii,self.shrinkjj
class CPW(Electrode):
    def __init__(self, gap, hot, material, layers, ylayer, gridnum, gridsize=500, modeargs=None):
        self.gap,self.hot,self.ylayer,self.gridsize = gap,hot,ylayer,gridsize
        super().__init__(material, layers, gridx=[-gridsize, -gap-hot/2, -hot/2, hot/2, hot/2+gap, gridsize], gridy=[-gridsize]+ylayer+[gridsize], gridnum=gridnum, modeargs=modeargs)
class CPS(Electrode):
    def __init__(self, gap, hot, material, layers, ylayer, gridnum, gridsize=500, modeargs=None):
        self.gap,self.hot,self.gnd,self.ylayer,self.gridsize = gap,hot,hot,ylayer,gridsize
        super().__init__(material, layers, gridx=[-gridsize, -hot-gap/2, -gap/2, gap/2, gap/2+hot, gridsize], gridy=[-gridsize]+ylayer+[gridsize], gridnum=gridnum, modeargs=modeargs)
    def targethot(self,z=50,hots=None,extrapolate='log'):
        hots = hots if hots is not None else [self.hot*2,self.hot,self.hot/2]
        zs = [CPS(self.gap, h, self.material, self.layers, self.ylayer, self.gridnum, self.gridsize, self.modeargs).Z for h in hots]
        print('        hot,Z:',hots,zs)
        return Wave(hots,zs)(z,extrapolate=extrapolate)
    def targetgap(self,z=50,gaps=None,extrapolate='log'):
        gaps = gaps if gaps is not None else [self.gap/2,self.gap,self.gap*2]
        zs = [CPS(g, self.hot, self.material, self.layers, self.ylayer, self.gridnum, self.gridsize, self.modeargs).Z for g in gaps]
        print('        gap,Z:',gaps,zs)
        return Wave(gaps,zs)(z,extrapolate=extrapolate)
        # return [Electrode(**{k:v for k,v in self.args.items() if not k==gap},gap=g).Z for g in gaps] # print('self.args',self.args)
        # return [Electrode(self.material, self.layers, self.gridx, self.gridy, self.gridnum, self.stretch, self.margin, self.modeargs).Z for g in gaps] # print('self.args',self.args)
class XcutElectrode(Electrode):
    def __init__(self, gap,hot,metal,film,substrate, buffer=None, gnd=None, gridsize=500, gridnum=36000, modeargs=None):
        if gnd:
            material, layers = {1:substrate,2:'TFLN',3:'air',4:'hot',5:'ground'}, [1,2,[3,5,3,4,3,5,3],3]
            gridx = [-gridsize,-gnd-gap-hot/2,-gap-hot/2,-hot/2,hot/2,gap+hot/2,gnd+gap+hot/2,gridsize]
            gridy = [-gridsize,-film,0,metal,gridsize]
            if buffer:
                material, layers = {1:substrate,2:'TFLN',3:'sio2',4:'air',5:'hot',6:'ground'}, [1,2,3,[4,6,4,5,4,6,4],4]
                gridy = [-gridsize,-film,0,buffer,buffer+metal,gridsize]
        else:
            material, layers = {1:substrate,2:'TFLN',3:'air',4:'hot',5:'ground'}, [1,2,[5,3,4,3,5],3]
            print(gridsize,gap,hot,film,0,metal,gridsize)
            gridx, gridy = [-gridsize,-gap-hot/2,-hot/2,hot/2,gap+hot/2,gridsize], [-gridsize,-film,0,metal,gridsize]
            if buffer:
                material, layers = {1:substrate,2:'TFLN',3:'sio2',4:'air',5:'hot',6:'ground'}, [1,2,3,[6,4,5,4,6],4]
                gridy = [-gridsize,-film,0,buffer,buffer+metal,gridsize]
        # modeargs = modeargs if modeargs is not None else {'λ':1550,'width':6,'depth':0.8,'ape':2.5,'rpe':0,'xcut':1,'res':0.2,'sell':'ln','plotmode':0,'cachelookup':1,'verbose':1}
        # super().__init__(material, layers, gridx, gridy, gridnum, modeargs=modeargs)
        Electrode.__init__(self, material, layers, gridx, gridy, gridnum, modeargs=modeargs)
        self.gap,self.hot,self.metal,self.film,self.substrate,self.buffer,self.gnd,self.gridnum,self.gridsize,self.modeargs = gap,hot,metal,film,substrate,buffer,gnd,gridnum,gridsize,modeargs
    def targethot(self,z=50,hots=None,extrapolate='log'):
        hots = hots if hots is not None else [self.hot*2,self.hot,self.hot/2]
        # zs = [XcutElectrode(self.gap, h, self.metal, self.film, self.substrate, self.buffer, self.gnd, self.gridsize, self.gridnum, self.modeargs).Z for h in hots]
        zs = [self.__class__(self.gap, h, self.metal, self.film, self.substrate, self.buffer, self.gnd, self.gridsize, self.gridnum, modeargs=self.modeargs).Z for h in hots]
        print('        hot,Z:',hots,zs)
        return Wave(hots,zs)(z,extrapolate=extrapolate)
    def targetgap(self,z=50,gaps=None,extrapolate='log'):
        gaps = gaps if gaps is not None else [self.gap/2,self.gap,self.gap*2]
        # zs = [XcutElectrode(g, self.hot, self.metal, self.film, self.substrate, self.buffer, self.gnd, self.gridsize, self.gridnum, self.modeargs).Z for g in gaps]
        zs = [self.__class__(g, self.hot, self.metal, self.film, self.substrate, self.buffer, self.gnd, self.gridsize, self.gridnum, modeargs=self.modeargs).Z for g in gaps]
        print('        gap,Z:',gaps,zs)
        return Wave(gaps,zs)(z,extrapolate=extrapolate)
    def legendtext(self,xoffset=None):
        return f"{self.gap}-{self.hot}-{self.gap} gap-hot-gap\n" + super().legendtext(xoffset)
        # f"Z={self.Z:.1f}Ω\nC={self.C:.2f}pF/cm, L={self.L:.2f}nH/cm\n$n_{{IR}}$={self.nir():.2f}, $n_{{RF}}$={self.nrf:.2f}\nVp={self.vpi():.1f}V·cm at x={self.bestoffset():.1f}µm"
    def savetext(self):
        return f'{self.gap}-{self.hot}-{self.gap} gap-hot-gap, {self.hot}µm hot, {self.metal}µm metal'+('' if self.buffer is None else f', {self.buffer}µm buffer')+f', {self.film}µm LN on {self.substrate}'
class TflnRidgeElectrode(Electrode):
    def __init__(self, gap,metal,film,substrate, gridsize=500, gridnum=36000, modeargs=None):
        material, layers = {1:substrate,2:'TFLN',3:'air',4:'hot',5:'ground'}, [1,[3,4,2,5,3],3]
        gridx, gridy = [-gridsize,-metal-gap/2,-gap/2,gap/2,metal+gap/2,gridsize], [-gridsize,-film,0,gridsize]
        modeargs = modeargs if modeargs is not None else {'λ':1550,'width':6,'depth':0.8,'ape':2.5,'rpe':0,'xcut':1,'res':0.2,'sell':'ln','plotmode':0,'cachelookup':1,'verbose':1}
        super().__init__(material, layers, gridx, gridy, gridnum, modeargs=modeargs)
        self.gap,self.metal,self.film,self.substrate = gap,metal,film,substrate
    def materialtext(self):
        pos = 2
        return [{'x':self.gridx[pos+1]/2-0.75*self.margin,'y':min(self.gridy[i+1],self.gridy[-2]+self.margin)-self.margin/2,'s':self.material[ks[pos]]} for i,ks in enumerate(self.filledoutlayers())]
    def legendtext(self,xoffset=None):
        return Electrode.legendtext(self,xoffset)
    def savetext(self):
        return f'{self.gap}µm gap, {self.metal}µm metal'+f', {self.film}µm LN on {self.substrate}'
class AluminaInterconnect(Electrode):
    def __init__(self,gap=310,hot=140,width=10000,thickness=125,gridsize=5000,gridnum=36000):
        material = {1:'air', 2:'alumina', 3:'hot', 4:'ground'}
        layers = [4,[1,2,2,2,2,2,1],[1,4,1,3,1,4,1],1]
        gridx = [-gridsize, -width/2, -gap-hot/2, -hot/2, hot/2, hot/2+gap, width/2, gridsize]
        if 2*gridsize<=width or gap-hot/2>=width/2:
            layers = [4,2,[4,1,3,1,4],1]
            gridx = [-gridsize, -gap-hot/2, -hot/2, hot/2, hot/2+gap, gridsize]
        gridy = [-gridsize, -thickness, 0, 10, gridsize] # bottom to top
        super().__init__(material, layers, gridx, gridy, gridnum)
        self.gap,self.hot,self.width,self.thickness = gap,hot,width,thickness
    def savetext(self):
        return f'{self.gap}-{self.hot}-{self.gap} gap-hot-gap, {self.width}µm width, {self.thickness}µm thickness, alumina interconnect'
class SetbackElectrode(XcutElectrode):
    def __init__(self, gap,hot,metal,film,substrate,buffer,gnd, buffersetback=0, gridsize=500, gridnum=36000, modeargs=None):
        material = {1:substrate,2:'TFLN',3:'sio2',4:'air',5:'hot',6:'ground'}
        gridy = [-gridsize,-film,0,buffer,buffer+metal,gridsize]
        # print('buffersetback',buffersetback)
        if buffersetback:
            layers =  [1,2,[3,3,6,4,5,3,5,4,6,3,3],[4,6,6,4,5,5,5,4,6,6,4],4]
            def gx(g,hh,gnd,gs,s): # hh=halfhot
                return [-gs,-gnd-g-hh, -g-hh-s ,-g-hh, -hh ,-hh+s,hh-s, hh ,g+hh, g+hh+s ,gnd+g+hh,gs]
            gridx = gx(gap,hot/2,gnd,gridsize,buffersetback)
            # gridx = [-gridsize,-gnd-gap-hot/2, -gap-hot/2 ,-gap-hot/2, -hot/2 ,-hot/2,hot/2, hot/2 ,gap+hot/2, gap+hot/2 ,gnd+gap+hot/2,gridsize]
        else:
            layers =  [1,2,[3,3,4,3,4,3,3],[4,6,4,5,4,6,4],4]
            gridx = [-gridsize,-gnd-gap-hot/2,-gap-hot/2,-hot/2,hot/2,gap+hot/2,gnd+gap+hot/2,gridsize]
        # print('SetbackElectrode\nmaterial',material, '\nlayers',layers, '\ngridx',gridx, '\ngridy',gridy, '\ngridnum',gridnum,'\nmodeargs',modeargs)
        Electrode.__init__(self, material, layers, gridx, gridy, gridnum, modeargs=modeargs)
        self.gap,self.hot,self.metal,self.film,self.substrate,self.buffer,self.gnd,self.buffersetback,self.gridnum,self.gridsize,self.modeargs = gap,hot,metal,film,substrate,buffer,gnd,buffersetback,gridnum,gridsize,modeargs
    def targethot(self,z=50,hots=None,extrapolate='log'):
        hots = hots if hots is not None else [self.hot*2,self.hot,self.hot/2]
        zs = [self.__class__(self.gap, h, self.metal, self.film, self.substrate, self.buffer, self.gnd, self.buffersetback, self.gridsize, self.gridnum, modeargs=self.modeargs).Z for h in hots]
        print('        hot,Z:',hots,zs)
        return Wave(hots,zs)(z,extrapolate=extrapolate)
    def targetgap(self,z=50,gaps=None,extrapolate='log'):
        gaps = gaps if gaps is not None else [self.gap/2,self.gap,self.gap*2]
        zs = [self.__class__(g, self.hot, self.metal, self.film, self.substrate, self.buffer, self.gnd, self.buffersetback, self.gridsize, self.gridnum, modeargs=self.modeargs).Z for g in gaps]
        print('        gap,Z:',gaps,zs)
        return Wave(gaps,zs)(z,extrapolate=extrapolate)
class AbuttingRidgeCPS(Electrode):
    def __init__(self,width=2,hot=10,film=4,etch=3,metal=None,substrate='quartz',buffer=1.0,buffermaterial='SiO2',gridsize=500,gridnum=36000,modeargs=None,λ=1550):
        metal = metal if metal is not None else etch
        assert etch<=film and etch<=metal
        material = {1:substrate,2:'xcutLN',3:buffermaterial,4:'air',5:'hot',6:'ground'}
        gridx = [-gridsize, -width/2-buffer-hot, -width/2-buffer, -width/2, width/2, width/2+buffer, width/2+buffer+hot, gridsize]
        gridy = [-gridsize,-film,-etch,0,gridsize] if metal==etch else [-gridsize,-film,-etch,0,metal-etch,gridsize]
        layers = [1,2,[4,6,3,2,3,5,4],4] if metal==etch else [1,2,[4,6,3,2,3,5,4],[4,6,4,4,4,5,4],4]
        if film==etch:
            gridy = [-gridsize,-film,0,gridsize] if metal==etch else [-gridsize,-film,0,metal-etch,gridsize]
            layers = [1,[4,6,3,2,3,5,4],4] if metal==etch else [1,[4,6,3,2,3,5,4],[4,6,4,4,4,5,4],4]
        modeargs = modeargs if modeargs is not None else dict(λ=λ,width=width,depth=film,ape=etch,rpe=0,xcut=1,
            res=0.1,limits=(-10,10,-10,2),sell='lnridgez',method='isotropic',cachelookup=1,verbose=1,nummodes=5)
        Electrode.__init__(self, material, layers, gridx, gridy, gridnum, modeargs=modeargs)
        self.gap = width+2*buffer
        self.hot,self.metal,self.film,self.substrate,self.buffer,self.buffermaterial = hot,metal,film,substrate,buffer,buffermaterial
        self.gridnum,self.gridsize,self.modeargs = gridnum,gridsize,modeargs
        self.width,self.etch,self.λ = width,etch,λ
    def materialtext(self):
        return []
    def legendtext(self,xoffset=0):
        s = f'Z={self.Z:.1f}Ω\nC={self.C:.2f}pF/cm\nL={self.L:.2f}nH/cm\n$n_{{RF}}$={self.nrf:.2f}'
        if self.modeargs is None:
            return s
        return s + f', $n_{{IR}}$={self.nir():.2f}\nVπ={self.vpi(xoffset=0):.2f}V·cm' + bool(xoffset)*f' at x={self.bestoffset():.1f}µm'
class RidgeCPS(Electrode):
    def __init__(self, gap=5,hot=10,metal=5,width=3,film=3,etch=2,substrate='quartz',buffer=0.2, gridsize=500, gridnum=36000, modeargs=None,λ=1550):
        assert buffer < etch < film < metal, f"buffer < etch < film < metal, {buffer} < {etch} < {film} < {metal}"
        material = {1:substrate,2:'xcutLN',3:'SiO2',4:'air',5:'hot',6:'ground'}
        layers = [1,2,[4,3,4,2,4,3,4],[4,6,4,2,4,5,4],[4,6,4,4,4,5,4],4]
        # xs = [width,gap,gap+2*hot]
        gridx = [-gridsize, -gap/2-hot, -gap/2, -width/2, width/2, gap/2, gap/2+hot, gridsize]
        gridy = [-gridsize,-film,-etch,-etch+buffer,0,-etch+buffer+metal,gridsize]
        modeargs = modeargs if modeargs is not None else dict(λ=λ,width=width,depth=film,ape=etch,rpe=0,xcut=1,
            res=0.2,limits=(-10,10,-10,2),sell='lnridgez',method='isotropic',cachelookup=1,verbose=1)
        Electrode.__init__(self, material, layers, gridx, gridy, gridnum, modeargs=modeargs)
        self.gap,self.hot,self.metal,self.film,self.substrate,self.buffer = gap,hot,metal,film,substrate,buffer
        self.gridnum,self.gridsize,self.modeargs = gridnum,gridsize,modeargs
        self.width,self.etch,self.λ = width,etch,λ
    def materialtext(self):
        return []
    def legendtext(self,xoffset=None):
        assert 0==xoffset or xoffset is None
        xoffset = 0
        s = f'Z={self.Z:.1f}Ω\nC={self.C:.2f}pF/cm\nL={self.L:.2f}nH/cm\n$n_{{RF}}$={self.nrf:.2f}'
        if self.modeargs is None:
            return s
        return s + f', $n_{{IR}}$={self.nir():.2f}\nVp={self.vpis()(xoffset):.1f}V·cm'
    def savetext(self):
        return f'{self.hot:g}-{self.gap:g}-{self.hot:g} electrode, {self.λ}nm {self.width:g}x{self.etch:g}µm ridge on {self.film:g}µm TFLN on {self.substrate}'+f', {self.buffer:g}µm buffer'*(1e-2<self.buffer)
    def plotmode(self,xoffset=None,savetext='',xlim=None,ylim=None,materialtext=None,**kwargs):
        xoffset = xoffset if xoffset is not None else self.bestoffset()
        xlim,ylim = self.plotlimits(xlim,ylim,aspect='aspect' in kwargs)
        ii = self.md().ee**2
        ii.xs = ii.xs + xoffset
        plot(contourf=self.potential,contour=ii,xlabel='x (µm)',ylabel='y (µm)',xlim=xlim,ylim=ylim,
            lines=self.plotlines(),texts=materialtext if materialtext is not None else self.materialtext(),
            legendtext=self.legendtext(xoffset),corner='upper right',
            save=(savetext if savetext else self.savetext()),**kwargs)

class RidgeElectrode(Electrode):
    def __init__(self, gap,hot,metal,film,substrate, buffer=None, buffer2=None, gridsize=500, gridnum=36000, modeargs=None):
        material, layers = {1:substrate,2:'LN',3:'SiO2',4:'epoxy',5:'hot',6:'ground'}, [1,[4,4,2,4,4],[6,4,5,4,6],3]
        gridx = [-gridsize,-gap-hot/2,-hot/2,hot/2,gap+hot/2,gridsize]
        gridy = [-gridsize,-film,0,metal,gridsize]
        if buffer:
            material, layers = {1:substrate,2:'LN',3:'SiO2',4:'epoxy',5:'hot',6:'ground'}, [1,[4,4,2,4,4],4,[6,4,5,4,6],3]
            gridx = [-gridsize,-gap-hot/2,-hot/2,hot/2,gap+hot/2,gridsize]
            gridy = [-gridsize,-film,0,buffer,buffer+metal,gridsize]
        if buffer2:
            material, layers = {1:substrate,2:'LN',3:'SiO2',4:'epoxy',5:'hot',6:'ground'}, [1,3,[4,4,2,4,4],4,[6,4,5,4,6],3]
            gridx = [-gridsize,-gap-hot/2,-hot/2,hot/2,gap+hot/2,gridsize]
            gridy = [-gridsize,-buffer2-film,-film,0,buffer,buffer+metal,gridsize]
        modeargs = modeargs if modeargs is not None else {'λ':1550,'width':9,'depth':9,'ape':9,'rpe':0,'xcut':0,
            'res':0.2,'sell':'mglnridge','plotmode':0,'cachelookup':1,'verbose':1}
        Electrode.__init__(self, material, layers, gridx, gridy, gridnum, modeargs=modeargs)
        self.gap,self.hot,self.metal,self.film,self.substrate,self.buffer,self.buffer2,self.gridnum,self.gridsize,self.modeargs = gap,hot,metal,film,substrate,buffer,buffer2,gridnum,gridsize,modeargs
    def savetext(self):
        return f'{self.gap}-{self.hot}-{self.gap} electrode, {self.film}µm LN on {self.substrate}'
    def materialtext(self):
        labellayers = [0,2,5] if self.buffer2 else [0,1,4]
        return [{'x':self.gridx[1]-self.margin/2, 'y':min(self.gridy[i+1],self.gridy[-2]+self.margin)-self.margin/2,
               's':self.material[ks[0]]} for i,ks in enumerate(self.filledoutlayers()) if i in labellayers] # if 'hot' not in [self.material[ks[j]] for j in range(len(ks))]]
    def legendtext(self,_):
        s = f'Z={self.Z:.1f}Ω\nC={self.C:.2f}pF/cm\nL={self.L:.2f}nH/cm\n$n_{{RF}}$={self.nrf:.2f}'
        if self.modeargs is None:
            return s
        return s + f', $n_{{IR}}$={self.nir():.2f}\nVp={self.vpis()(0):.1f}V·cm'
class OverhangRidgeElectrode(Electrode):
    def __init__(self, gap,hot,metal,width,film,substrate,crush,buffer2, submount='SiO2', gridsize=500, gridnum=36000, modeargs=None, deepdice=0):
        assert width<hot
        material, layers = {1:substrate,2:'LN',3:'SiO2',4:'epoxy',5:submount,6:'hot',7:'ground'}, [1,3,[4,4,4,2,4,4,4],4,[7,4,6,6,6,4,7],5]
        gridx = [-gridsize,-gap-hot/2,-hot/2,-width/2,width/2,hot/2,gap+hot/2,gridsize]
        gridy = [-gridsize,-buffer2-film,-film,0,crush,crush+metal,gridsize]
        if deepdice:
            layers = [1,[4,4,4,1,4,4,4],[4,4,4,3,4,4,4],[4,4,4,2,4,4,4],4,[7,4,6,6,6,4,7],5]
            gridy = [-gridsize,-deepdice-buffer2-film,-buffer2-film,-film,0,crush,crush+metal,gridsize]

        modeargs = modeargs if modeargs is not None else {'λ':1550,'width':film,'depth':film,'ape':film,'rpe':0,'xcut':0,
            'res':0.2,'sell':'mglnridge','plotmode':0,'cachelookup':1,'verbose':1}
        Electrode.__init__(self, material, layers, gridx, gridy, gridnum, modeargs=modeargs)
        self.gap,self.hot,self.metal,self.film,self.substrate,self.crush,self.buffer2,self.gridnum,self.gridsize,self.modeargs = gap,hot,metal,film,substrate,crush,buffer2,gridnum,gridsize,modeargs
    def savetext(self):
        return f'{self.gap}-{self.hot}-{self.gap} electrode, {self.film}µm LN on {self.substrate}'
    def materialtext(self):
        labellayers = [0,2,5] if self.buffer2 else [0,1,4]
        return [{'x':self.gridx[1]-self.margin/2, 'y':min(self.gridy[i+1],self.gridy[-2]+self.margin)-self.margin/2,
               's':self.material[ks[0]]} for i,ks in enumerate(self.filledoutlayers()) if i in labellayers] # if 'hot' not in [self.material[ks[j]] for j in range(len(ks))]]
    def legendtext(self,_=''):
        s = f'Z={self.Z:.1f}Ω\nC={self.C:.2f}pF/cm\nL={self.L:.2f}nH/cm\n$n_{{RF}}$={self.nrf:.2f}'
        if self.modeargs is None:
            return s
        return s + f', $n_{{IR}}$={self.nir():.2f}\nVp={self.vpis()(0):.1f}V·cm'
class XcutSubmountCPW(Electrode):
    def __init__(self,gap=20,hot=6.5,metal=3.5,buffer=0.3,film=4,etch=2,wgwidth=7,wgoffset=-10,substrate='xcutln',gridsize=500,gridnum=36000,modeargs=None):
        material = {1:'quartz',2:substrate,3:'',4:'quartz',5:'hot',6:'ground'}
        ridgelayer = Wave.fromtiles([3,2,3],[-gridsize,wgoffset-wgwidth/2,wgoffset+wgwidth/2,gridsize])
        eleclayer = Wave.fromtiles([6,3,5,3,6],[-gridsize,-gap-hot/2,-hot/2,hot/2,gap+hot/2,gridsize])
        ridgelayer,eleclayer = ridgelayer.dualmergex(eleclayer)
        assert np.all(ridgelayer.x==eleclayer.x)
        ry,rx = ridgelayer.wave2tile()
        ey,ex = eleclayer.wave2tile()
        assert np.all(rx==ex)
        layers = [1,2,[int(x) for x in ry],3,[int(x) for x in ey],1]
        gridx,gridy = ex,[-gridsize,-film,-etch,0,buffer,buffer+metal,gridsize]
        modeargs = modeargs if modeargs is not None else dict(λ=1550,width=wgwidth,depth=film,ape=etch,rpe=0,xcut=1,
            res=0.2,limits=(-10,10,-10,2),sell='lnridgey',method='isotropic',cachelookup=1,verbose=0)
        Electrode.__init__(self, material, layers, gridx, gridy, gridnum, modeargs=modeargs)
        self.gap,self.hot,self.metal,self.buffer,self.film,self.etch,self.wgwidth,self.wgoffset,self.substrate,self.gridsize,self.gridnum,self.modeargs = gap,hot,metal,buffer,film,etch,wgwidth,wgoffset,substrate,gridsize,gridnum,modeargs
    def vpi(self,xoffset=None,dead=False,db=0,length=10): # units of V·cm by default
        C = 10**(abs(db)/20) * 10/length
        xoffset = xoffset if xoffset is not None else self.wgoffset
        return C*self.vpis(dead=dead,xs=np.array([xoffset])).atx(xoffset,monotonicx=False)
    def plotmode(self,xoffset=None,savetext='',xlim=None,ylim=None,materialtext=None,**kwargs):
        xoffset = xoffset if xoffset is not None else self.wgoffset
        return super().plotmode(xoffset=xoffset,savetext=savetext,xlim=xlim,ylim=ylim,materialtext=materialtext,**kwargs)
    def legendtext(self,xoffset=None,loss=False):
        return super().legendtext(xoffset=xoffset,loss=loss)

def electricsolve(material,gridsubs,gridx,gridy,gridnum,stretch=20,xstretch=None,ystretch=None,cachelookup=True):
    if cachelookup: return joblib.Memory(cachefolder, verbose=0).cache(electricsolve)(material,gridsubs,gridx,gridy,gridnum,stretch,xstretch,ystretch,cachelookup=False)
    assert all([x0<x1 for x0,x1 in zip(gridx[:-1],gridx[1:])]) and all([y0<y1 for y0,y1 in zip(gridy[:-1],gridy[1:])]), f'invalid grid, gridx:{gridx} gridy:{gridy}'
    folder='c:/temp/pyelmer/'
    clearfolder(folder)
    writegrd(gridx,gridy,gridsubs,gridnum,folder,stretch=stretch,xstretch=xstretch,ystretch=ystretch)
    writesif(material,folder,vacuum=True)
    runelmer(folder)
    cL = loadvalues(folder)['capacitance']
    writesif(material,folder)
    runelmer(folder)
    cC = loadvalues(folder)['capacitance']
    c,e0 = 299792458,8.8541878128e-12
    L,Z0 = 1/(e0*cL*c**2)*1e9/100, 1/(c*cL*e0)
    C,Z,nrf = cC*e0*1e12/100, np.sqrt((L*1e-9*100)/(cC*e0)), np.sqrt((L*1e-9*100)*(cC*e0)*c**2)
    xx,yy = loadgidmsh(folder)
    nx,ny = len(xx),len(yy)
    v,ex,ey,eed = [loadgidres(folder,s) for s in ['potential','electric field 1','electric field 2','electric energy density']] # assert v.shape==ex.shape==ey.shape==eed.shape
    v,ey,ex,eed = [Wave2D(w.reshape(ny,nx).T,xs=xx,ys=yy) for w in [v,ey,ex,eed]]
    dvdy = Wave2D(np.diff(v,axis=1)/np.diff(v.ys)[None,:], v.xs, v.ys[1:]/2+v.ys[:-1]/2)
    dvdx = Wave2D(np.diff(v,axis=0)/np.diff(v.xs)[:,None], v.xs[1:]/2+v.xs[:-1]/2, v.ys)
    return {'L':L,'C':C,'Z':Z,'Z0':Z0,'nrf':nrf,'v':v,'dvdy':dvdy,'ey':ey,'dvdx':dvdx,'ex':ex,'xx':xx,'yy':yy} # print(f'L={L:.3f}nH/cm, C={C:.3f}pF/cm, Z={Z:.2f}Ω, Z0={Z0:.2f}Ω, nrf={nrf:.3f}')
def runelmer(folder,elmerfolder='c:/elmer/bin/'):
    import subprocess, glob, shutil, os
    def run(command,logfile):
        log = open(logfile, 'w')
        # print('    running:',command) # comment or uncomment won't affect cache
        process = subprocess.Popen(command.split(), cwd=folder, stdout=log, stderr=subprocess.PIPE)
        out,err = process.communicate()
        if err: print('elmer.py error:',err)
    if elmerverbose: print('    running:',elmerfolder+'elmergrid 1 2 elmer.grd',',',elmerfolder+'elmersolver')
    run(elmerfolder+'elmergrid 1 2 elmer.grd', folder+'grd.txt')
    run(elmerfolder+'elmersolver', folder+'out.txt')
def loadvalues(folder):
    def names():
        with open(folder+'scalars.dat.names','r') as f:
            if not f.readline().startswith('File started'):
                for i in range(5): f.readline()
            assert f.readline().strip().startswith('Variables in columns of matrix')
            return [line.split(':')[-1].strip() for line in f] # e.g. line='   1: res: electric energy'
    def values():
        with open(folder+'scalars.dat','r') as f:
            return [float(s) for s in f.readline().split()] # e.g. f.readline()='   9.685030985213E+000   1.000000000000E+000   1.937006197043E+001'
    return {k:v for k,v in zip(names(),values())}
def loadflavia(file,label,name=''):
    with open(file,'r') as f:
        if name:
            while not f.readline().startswith(f'ComponentNames "{name}"'): pass
        while not f.readline().startswith(label.title()): pass
        while 1:
            line = f.readline().strip()
            if line.startswith('end '+label): return
            yield [-1 if '******'==s else float(s) for s in line.split()]
            #case.flavia.msh:
            # 999998 -2.7346126E+002  9.9047217E+002  0.0000000E+000
            # 999999 -2.7240016E+002  9.9047217E+002  0.0000000E+000
            # ****** -2.7135145E+002  9.9047217E+002  0.0000000E+000
def loadgidmsh(folder):
    def gridxy(cx,cy):
        def unique(s): # ordered remove duplicates
            from collections import OrderedDict
            return list(OrderedDict.fromkeys(s))
        return np.array(unique(cx)),np.array(unique(cy))
    try:
        cc = [*loadflavia(folder+'case.flavia.msh','coordinates')]
    except FileNotFoundError:
        cc = [*loadflavia(folder+'elmer/case.flavia.msh','coordinates')]
    cx,cy = np.array([x for n,x,y,z in cc]),np.array([y for n,x,y,z in cc])
    return gridxy(cx,cy)
def loadgidres(folder,name):
    try:
        cc = [*loadflavia(folder+'case.flavia.res','values',name)]
    except FileNotFoundError:
        cc = [*loadflavia(folder+'elmer/case.flavia.res','values',name)]
    return np.array([z for n,z in cc])
def filesdiff(filea,fileb):
    from difflib import unified_diff
    with open(filea, 'r') as a, open(fileb, 'r') as b:
        sa,sb = a.readlines(), b.readlines()
        for line in unified_diff(sa,sb,n=0):
            print(line.strip())
def writegrd(gridx,gridy,gridsubs,gridnum,folder,stretch=1,xstretch=None,ystretch=None):
    def idmax():
        return max([(max(row) if hasattr(row,'__len__') else row)for row in gridsubs])
    nx,ny = len(gridx),len(gridy)
    hot,ground = idmax()-1,idmax()  # assuming ground has highest id, hot next highest
    xstretch = xstretch if xstretch is not None else [1./stretch]+[1 for n in range(nx-3)]+[stretch]
    ystretch = ystretch if ystretch is not None else [1./stretch]+[1 for n in range(ny-3)]+[stretch]
    assert gridsubs.shape==(ny-1,nx-1), f'gridsubs size does not match gridx,gridy: gridsubs{gridsubs} gridx{gridx} gridy{gridy}'
    assert len(xstretch)==nx-1 and len(ystretch)==ny-1, f'xstretch,ystretch not length {nx-1},{ny-1}: {xstretch},{ystretch}'
    with open(folder+'elmer.grd','w') as f:
        f.write( '#####  ElmerGrid input file for structured grid generation  ######\n')
        f.write( 'Version = 210903\n')
        f.write( 'Coordinate System = Cartesian 2D\n')
        f.write(f'Subcell Divisions in 2D = {nx-1} {ny-1}\n')
        f.write( 'Subcell Limits 1 = ' +' '.join([f'{x:g}' for x in gridx])+'\n')
        f.write( 'Subcell Limits 2 =  '+' '.join([f'{y:g}' for y in gridy])+'\n')
        f.write( 'Material Structure in 2D\n')
        for row in gridsubs:
            f.write( '    '+'    '.join([f'{k}' for k in row])+' \n') # f.write( '    8    4    7    4    8 \n')
        f.write( 'End\n')
        f.write(f'Materials Interval = 1 {ground}\n')
        f.write( 'Boundary Definitions\n')
        f.write( '# type     out      int \n')
        for i in range(1,hot):
            f.write(f'  1        {hot}        {i}        1 \n') # f.write( '  1        7        1        1 \n')
        for i in range(1,hot):
            f.write(f'  2        {ground}        {i}        1 \n') # f.write( '  2        8        6        1 \n')
        for i in range(1,hot):
            f.write(f'  {i+2}        0        {i}        1 \n') # f.write( '  3        0        1        1 \n')
        f.write( 'End\n')
        f.write( 'Numbering = Horizontal\n')
        f.write( 'Element Degree = 1\n')
        f.write( 'Element Innernodes = False\n')
        f.write( 'Triangles = False\n')
        f.write(f'Surface Elements = {gridnum}\n')
        f.write(f'Element Ratios 1 = '+' '.join([f'{x:g}' for x in xstretch])+'\n') # f.write( 'Element Ratios 1 = 0.05 1 1 1 20\n')
        f.write(f'Element Ratios 2 = '+' '.join([f'{y:g}' for y in ystretch])+'\n')
        f.write( 'Element Densities 1 = '+' '.join([f'1' for x in xstretch])+'\n') # f.write( 'Element Densities 1 = 1 1 1 1 1\n')
        f.write( 'Element Densities 2 = '+' '.join([f'1' for x in ystretch])+'\n') # f.write( 'Element Densities 2 = 1 1 1 1 1 1 1\n')
def writesif(material,folder,vacuum=False):
    dielectrics = {k:v for k,v in material.items() if v not in ['hot','ground']}
    with open(folder+'ELMERSOLVER_STARTINFO','w') as f:
        f.write('case.sif\n1\n')
    with open(folder+'case.sif','w') as f:
        f.write('Header\n')
        f.write('  CHECK KEYWORDS Warn\n')
        f.write('  Mesh DB "." "./elmer"\n')
        f.write('  Include Path ""\n')
        f.write('  Results Directory "."\n')
        f.write('End\n')
        f.write('\n')
        f.write('Simulation\n')
        f.write('  Max Output Level = 4\n')
        f.write('  Coordinate System = Cartesian\n')
        f.write('  Coordinate Mapping(3) = 1 2 3\n')
        f.write('  Simulation Type = Steady state\n')
        f.write('  Steady State Max Iterations = 1\n')
        f.write('  Output Intervals = 1\n')
        f.write('  Timestepping Method = BDF\n')
        f.write('  BDF Order = 1\n')
        f.write('  Solver Input File = case.sif\n')
        f.write('  Output File = case.result\n')
        f.write('  Post File = case.ep\n')
        f.write('End\n')
        f.write('\n')
        f.write('Constants\n')
        f.write('  Gravity(4) = 0 -1 0 9.82\n')
        f.write('  Stefan Boltzmann = 5.67e-08\n')
        f.write('  Permittivity of Vacuum = 1\n')
        f.write('  Boltzmann Constant = 1.3807e-23\n')
        f.write('  Unit Charge = 1.602e-19\n')
        f.write('End\n')
        f.write('\n')
        for k in dielectrics:
            f.write(f'Body {k}\n')
            f.write(f'  Target Bodies(1) = {k}\n')
            f.write(f'  Name = "Body {k}"\n')
            f.write(f'  Equation = 1\n')
            f.write(f'  Material = {k}\n')
            f.write(f'End\n')
            f.write(f'\n')
        f.write('Solver 1\n')
        f.write('  Equation = Electrostatics\n')
        f.write('  Calculate Electric Energy = True\n')
        f.write('  Calculate Electric Field = True\n')
        f.write('  Variable = -dofs 1 Potential\n')
        f.write('  Procedure = "StatElecSolve" "StatElecSolver"\n')
        f.write('  Exec Solver = Always\n')
        f.write('  Stabilize = True\n')
        f.write('  Bubbles = False\n')
        f.write('  Lumped Mass Matrix = False\n')
        f.write('  Optimize Bandwidth = True\n')
        f.write('  Steady State Convergence Tolerance = 1.0e-5\n')
        f.write('  Nonlinear System Convergence Tolerance = 1.0e-8\n')
        f.write('  Nonlinear System Max Iterations = 20\n')
        f.write('  Nonlinear System Newton After Iterations = 3\n')
        f.write('  Nonlinear System Newton After Tolerance = 1.0e-3\n')
        f.write('  Nonlinear System Relaxation Factor = 1\n')
        f.write('  Linear System Solver = Iterative\n')
        f.write('  Linear System Iterative Method = BiCGStab\n')
        f.write('  Linear System Max Iterations = 500\n')
        f.write('  Linear System Convergence Tolerance = 1.0e-8\n')
        f.write('  Linear System Preconditioning = ILU0\n')
        f.write('  Linear System ILUT Tolerance = 1.0e-3\n')
        f.write('  Linear System Abort Not Converged = False\n')
        f.write('  Linear System Residual Output = 1\n')
        f.write('  Linear System Precondition Recompute = 1\n')
        f.write('End\n')
        f.write('\n')
        f.write('Solver 2\n')
        f.write('Exec Solver = After All\n')
        f.write('Equation = SaveScalars\n')
        f.write('Procedure = "SaveData" "SaveScalars"\n')
        f.write('Filename = "scalars.dat"\n')
        f.write('End\n')
        f.write('\n')
        f.write('Solver 3\n')
        f.write('Exec Solver = after all\n')
        f.write('Equation = "result output"\n')
        f.write('Procedure = "ResultOutputSolve" "ResultOutputSolver"\n')
        f.write('Output File Name = "case"\n')
        f.write('Output Format = gid\n')
        f.write('Scalar Field 1 = potential\n')
        f.write('Scalar Field 2 = electric field 2\n')
        f.write('Scalar Field 3 = electric field 1\n')
        f.write('Scalar Field 4 = electric energy density\n')
        f.write('End\n')
        f.write('\n')
        f.write('Equation 1\n')
        f.write('  Name = "Electrostatics"\n')
        f.write('  Active Solvers(1) = 1\n')
        f.write('End\n')
        f.write('\n')
        for k,name in dielectrics.items():
            e = 1 if vacuum else relativepermittivity[name.lower()]
            f.write(f'Material {k}\n')
            f.write(f'  Name = "{name}"\n')
            if hasattr(e,'__len__'):
                # f.write(f'  Relative Permittivity(3) = {e[0]:g} {e[1]:g} {e[2]:g}\n')
                f.write(f'  Relative Permittivity({len(e)}) = {" ".join([f"{ei:g}" for ei in e])}\n')
            else:
                f.write(f'  Relative Permittivity = {e:g}\n')
            f.write(f'End\n')
            f.write(f'\n')
        f.write('Boundary Condition 1\n')
        f.write('  Target Boundaries(1) = 1\n')
        f.write('  Name = "Hot"\n')
        f.write('  Potential = 1\n')
        f.write('End\n')
        f.write('\n')
        f.write('Boundary Condition 2\n')
        f.write('  Target Boundaries(1) = 2\n')
        f.write('  Name = "Ground"\n')
        f.write('  Potential = 0\n')
        f.write('End\n')
def clearfolder(folder): # to remove old files before running to catch elmer.exe failure
    import os, send2trash
    path = os.path.abspath(folder)
    if not os.path.exists(path):
        print(path,'not found')
        os.mkdir(path)
    send2trash.send2trash(path)
    os.mkdir(path)
def rfbandwidth(Z,nrf,loss=1.0,lengthinmm=10,λ=1064,sell='ktpwg',fmax=40,df=0.01,za=50,zt=50,R=0,legendtext='',plot=True): # loss in dB/cm/√GHz
    # 20*log for optical response, 20*log for S11,S21 and RF loss, 10*log for |S11|² # print(Z,nrf,loss,lengthinmm,λ,sell,fmax,df,za,zt,legendtext,plot)
    lossdc = (10*0.5*R/Z/lengthinmm)*20/np.log(10); assert 0==R, 'double check *lengthinmm or /lengthinmm here?'
    # offset in dB due to dc resistance = (0.5*R/Z)*20/np.log(10)
    # αdc = 0.5*R/Z/lengthinmm # in mm⁻¹ = 10*0.5*R/Z/lengthinmm # in cm⁻¹
    # αac = np.log(10**(0.05*loss))*np.sqrt(f) # in cm⁻¹ with loss in dB/cm
    # αdc = np.log(10**(0.05*lossdc))
    nktp,length,c = index(λ,sell.replace('ridge',''),20),lengthinmm/1000,299792458
    x = np.array(wrange(0,fmax,df))
    x[0] += 1e-9
    def α(index):
        return np.log(10**(lossdc/20))*length/.01 + np.log(10**(loss/20))*np.sqrt(x)*length/.01 + 1j*length*2*np.pi*1e9*x/c*index
    def rfprop(za,z0,zt,alpha0): # za = input impedance, z0 = electrode impedance, zt = termination impedance
        g = (zt-z0)/(zt+z0)
        ge = np.exp(-alpha0)
        ga = ( 1/ge-g*ge )/( 1/ge+g*ge ) * za/z0
        ga = (1-ga)/(1+ga)
        gb = (1+g)*(1+ga)/( 1/ge+g*ge )
        # ga[0] = nan
        return g,ge,ga,gb
    def opprop(g,ge,ga,alpha1,alpha2):
        gg = (np.exp(+alpha1+1e-99)-1)/(+alpha1+1e-99)
        gg += g*(np.exp(-alpha2)-1)/(-alpha2)
        gg *= (1+ga)/( 1/ge+g*ge )
        return gg
    def dbwave(y,name=''):
        return Wave(20*np.log10(abs(y)),x,name)
    za, z0, zt = za+0j, Z+0j, zt+0j
    g,ge,ga,gb = rfprop(za,z0,zt,α(nrf))
    opticalcoprop = dbwave( opprop(g,ge,ga,α(nrf-nktp),α(nrf+nktp)),'optical, co-prop' )
    opticalcounterprop = dbwave( opprop(g,ge,ga,α(nrf+nktp),α(nrf-nktp)),'optical, counter-prop' )
    opticalvelocitymatched = dbwave( opprop(g,ge,ga,α(0),α(2*nrf)),'optical, velocity matched' )
    uu = dbwave( np.sqrt(ga*ga.conjugate() + gb*gb.conjugate()), 'RF, |S11|²+|S21|²' )
    s11 = dbwave(ga,'RF, |S11|')
    s21 = dbwave(gb,'RF, |S21|')
    opticallossless = dbwave( opprop( *rfprop(za,z0,zt,1j*α(nrf).imag)[:3], 1j*α(nrf-nktp).imag, 1j*α(nrf+nktp).imag),'optical, no loss' )
    def sinc(x):
        return np.sinc(x/np.pi)
    def bestlengthinmm(f):
        return 1000*299792458/(2*1e9*f*(nrf-nktp))
    opticallossless50ohm = dbwave( sinc(0.5*α(nrf-nktp).imag),'optical, no loss 50Ω' )
    opticallossless50ohmcounterprop = dbwave( sinc(0.5*α(nrf+nktp).imag),'optical, counter-prop no loss 50Ω' )
    opbestlen = dbwave( np.where(0.5*α(nrf-nktp).imag>pi/2, 1/(0.5*α(nrf-nktp).imag+1e-99), sinc(0.5*α(nrf-nktp).imag)),'optical, best length' ) # opticallossless50ohm with best length chosen, length = min( len, bestlength(f) )
    if plot: Wave.plots(opticalcoprop,opticalcounterprop,opticalvelocitymatched,opticallossless,
            opticallossless50ohm,opticallossless50ohmcounterprop,opbestlen,s11,s21,uu,
            seed=16,x='frequency (GHz)',y='response (dB)',ylim=((np.nanmin(opticalcounterprop)//10)*10,None),
            legendtext=(legendtext+'\n' if legendtext else '')+f'RF loss = {loss}dB/cm/√GHz',
            save='rfbandwidth plot')
    return dotdict({'coprop':opticalcoprop,'counterprop':opticalcounterprop,'velocitymatched':opticalvelocitymatched,'s11':s11,'s21':s21,'lossless':opticallossless,'lossless50ohm':opticallossless50ohm,'lossless50ohmcounterprop':opticallossless50ohmcounterprop,'opbestlen':opbestlen,'sums11s21':uu})
def skindepth(f,metal='gold'): # in µm, f in GHz
    µ0 = 4*pi*1e-13 # H/µm = V·s/A·µm = Ω·s/µm
    σ = conductivity[metal] # 1/Ω/µm
    return 1/np.sqrt(pi*f*1e9*µ0*σ)
def skinfreq(δ,metal='gold'): # in GHz, δ in µm
    µ0 = 4*pi*1e-13 # Ω·s/µm
    σ = conductivity[metal] # 1/Ω/µm
    return 1/(pi*δ**2*1e9*µ0*σ)
def skineffect(metalx,metaly,loss,metal='Cu'):
    rho = {'Cu':1.68e-8,'Au':2.44e-8}[metal] # Cu rho=168ohms*um^2/cm, Au rho=244ohms*um^2/cm
    perimeter = metalx+metalx+metaly+metaly
    skin = (rho*.01*1e6**2) / (np.log(10)*2*50*loss/20) / perimeter
    # skin depth = 2.06 um Cu, 2.50 um Au at f = 1 GHz // http://en.wikipedia.org/wiki/Skin_effect
    print( "loss(dB/cm):",loss,"alpha(1/cm):",np.log(10**(loss/20)),"  skindepth(um)",skin )
def mzresponse():
    ### EO modulator
    # E = E₀ exp[iω₀t + iβ sin ωt]
    # E = E₀ exp[iω₀t] Σ Jn(β) exp[inωt]
    # ωn = w₀ + n ω
    # In = Jn²(β)
    ### MZ interferometer
    # E = E₀ exp[iω₀t + iφ₀ + iβ₀ sin ωt] + E₁ exp[iω₀t + iφ₁ + iβ₁ sin ωt]
    # E = exp[iω₀t] Σ exp[inωt] (E₀ exp[iφ₀] Jn(β₀) + E₁ exp[iφ₁] Jn(β₁))
    # ωn = w₀ + n ω
    # In = E₀² Jn²(β₀) + E₁² Jn²(β₁) + 2 E₀ E₁ Jn(β₀) Jn(β₁) cos(φ₁-φ₀)
    def mzintensity(a,b,f,t): # a,b = vdc*pi/vpi,vac*pi/vpi
        return 0.5*(1+cos(a+b*sin(2*pi*f*t)))
    def coef(n,a,b):
        from scipy.special import jv
        return 0.5*(1+cos(a)*jv(0,b)) if 0==n else jv(n,b)*cos(a) if 0==n%2 else -jv(n,b)*sin(a)
    def mzexpanded(a,b,f,t,num=10):
        def term(n):
            wt = 1+0*t if 0==n else cos(n*2*pi*f*t) if 0==n%2 else sin(n*2*pi*f*t)
            return coef(n,a,b)*wt
        return sum(term(n) for n in range(num))
    wx = Wave(index=np.linspace(0,2,201))
    w0 = mzintensity(0.0*pi,0.5*pi,1,wx)
    w1 = mzintensity(0.5*pi,0.5*pi,1,wx)
    w2 = mzintensity(1.0*pi,0.5*pi,1,wx)
    w  = mzexpanded(0.5*pi,0.5*pi,1,wx)
    # Wave.plots(w0,w1,w2,w)
    def mzout(e0,e1,phi,db,f,t):
        return e0**2 + e1**2 + 2*e0*e1*cos(phi+db*sin(2*pi*f*t))
    def mzterm(n,e0,e1,phi,beta0,beta1):
        from scipy.special import jv
        return e0*jv(n,beta0) + e1*jv(n,beta1)*exp(1j*phi)
    print([abs(mzterm(n,1,1,3,-0.0,+1.0)) for n in [0,1]])
    print([abs(mzterm(n,1,1,3,-0.5,+0.5)) for n in [0,1]])
    print([abs(mzterm(n,1,1,3,-1.0,+0.0)) for n in [0,1]])
    def mzplot(e0,e1,phi,N=1):
        wx = np.linspace(0,10,101)
        w0s = [Wave(np.abs(mzterm(n,e0,e1,phi,-wx,0)),wx,n) for n in range(-N,N+1)]
        w1s = [Wave(np.abs(mzterm(n,e0,e1,phi,-wx/2,wx/2)),wx,n) for n in range(-N,N+1)]
        Wave.plots(*w0s,*w1s,groupsize=len(w0s))
    # mzplot(0.75,0.25,3)
    def mzsum(e0,e1,phi,b0,b1,f,t,N=10):
        return sum(exp(1j*2*pi*n*f*t)*mzterm(n,e0,e1,phi,b0,b1) for n in range(-N,N+1)).magsqr()
    def comparetest(r=0.5,b=1,f=1):
        ux = Wave(index=np.linspace(0,2,201))
        # u0 = mzout(r,1-r,   0,b,f,ux).rename(0)
        # u1 = mzout(r,1-r,pi/2,b,f,ux).rename(1)
        # u2 = mzout(r,1-r,  pi,b,f,ux).rename(2)
        u0 = mzsum(r,1-r,   0,-b/2,+b/2,f,ux).rename(0)
        u1 = mzsum(r,1-r,pi/2,-b/2,+b/2,f,ux).rename(1)
        u2 = mzsum(r,1-r,  pi,-b/2,+b/2,f,ux).rename(2)
        v0 = mzsum(r,1-r,   0,-0.9*b,0.1*b,f,ux).rename(0)
        v1 = mzsum(r,1-r,pi/2,-0.9*b,0.1*b,f,ux).rename(1)
        v2 = mzsum(r,1-r,  pi,-0.9*b,0.1*b,f,ux).rename(2)
        Wave.plots(u0,u1,u2,v0,v1,v2,l='000222')
    comparetest()
    # comparetest(r=0.25,b=2)
def sidebands(): # from sidebands.ipynb
    from scipy.special import jv
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy import pi,sin,cos,exp,abs
    def pmamplitude(n,beta): # sideband ampltude given n = sideband number, beta = pi*V/Vpi
        return jv(n,beta)
    def pmintensity(n,beta): # sideband intensity given n = sideband number, beta = pi*V/Vpi
        return pmamplitude(n,beta)**2
    beta,N = 1.5,10
    sum(pmintensity(n,beta) for n in range(-N,N+1))
    pmintensity(0,0)
    def pmsidebandplot(vpi,N=3):
        wx = np.linspace(0,2*vpi,41)
        for n in range(-N,N+1):
            plt.plot(wx, pmintensity(n,pi*wx/vpi), '.' if n<0 else '-', label=f'n = {n}')
        plt.legend(); plt.xlabel('drive voltage (V)'); plt.ylabel('$I_n$')
    # pmsidebandplot(vpi=1)
    def mzamplitude(n,E0,E1,phi,beta0,beta1): # Mach Zehnder sideband ampltude
        return E0*jv(n,beta0) + E1*jv(n,beta1)*exp(1j*phi)
    def mzintensity(n,E0,E1,phi,beta0,beta1): # Mach Zehnder sideband intensity
        return E0**2*jv(n,beta0)**2 + E1**2*jv(n,beta1)**2 + \
            E0*E1*jv(n,beta0)*jv(n,beta1)*cos(phi)
    def mzoutput(E0,E1,phi,db,f,t):
        return E0**2 + E1**2 + 2*E0*E1*cos(phi+db*sin(2*pi*f*t))
    def mzsum(E0,E1,phi,beta0,beta1,f,t,N=10):
        return abs(sum( exp(1j*2*pi*n*f*t) * mzamplitude(n,E0,E1,phi,beta0,beta1) for n in range(-N,N+1) ))**2
    def timeplot(E0,E1,phi,beta0,beta1,f=1,N=None):
        wx = np.linspace(0,2/f,201)
        if N is None:
            wy = mzoutput(E0,E1,phi,beta1-beta0,f,wx)
            plt.plot(wx, wy, '.', label='mzoutput')
        else:
            wy = mzsum(E0,E1,phi,beta0,beta1,f,wx,N=N)
            plt.plot(wx, wy, '-', label='mzsum')
        plt.legend(); plt.xlabel('time (s)'); plt.ylabel('intensity')
    # timeplot(0.5, 0.5, pi/2, -1, +1)
    # timeplot(0.5, 0.5, pi/2, -1, +1, N=10)
    # timeplot(0.25, 0.75, pi/4, beta0=-3, beta1=+0, N=10)
    # timeplot(0.25, 0.75, pi/4, beta0=-1, beta1=+2, N=10)
    # timeplot(0.25, 0.75, pi/4, beta0=-3, beta1=+0, N=2)
    # timeplot(0.25, 0.75, pi/4, beta0=-1, beta1=+2, N=2)
    def mzsidebandplot(E0,E1,phi,vpi0=1,vpi1=1,N=3):
        wx = np.linspace(0,2*vpi0,41)
        for n in range(-N,N+1):
            wy = mzintensity(n,E0,E1,phi,beta0=-pi*wx/vpi0,beta1=+pi*wx/vpi1)
            plt.plot(wx, wy, '.' if n<0 else '-', label=f'n = {n}')
        plt.legend(); plt.xlabel('drive voltage (V)'); plt.ylabel('$I_n$')
    # mzsidebandplot(0.5, 0.5, pi/2)
    # mzsidebandplot(0.2, 0.8, pi/2)
    # mzsidebandplot(0.5, 0.5, pi/4)
    # mzsidebandplot(0.5, 0.5, pi/8)
    # mzsidebandplot(0.5, 0.5, pi/4)
    # mzsidebandplot(0.5, 0.5, pi/8)
    # mzsidebandplot(0.5, 0.5, pi/18)
    # mzsidebandplot(0.5, 0.5, pi/8)
    # mzsidebandplot(0.2, 0.8, pi/8)
    # mzsidebandplot(0.5, 0.5, pi/8)
    # %history
    # plt.show()
    # mzsidebandplot(0.5, 0.5, pi/2); plt.show()
    def mzsidebandplotwithsum(E0,E1,phi,vpi0=1,vpi1=1,N=3):
        wx = np.linspace(0,2*vpi0,41)
        w0 = wx*0
        for n in range(-N,N+1):
            wy = mzintensity(n,E0,E1,phi,beta0=-pi*wx/vpi0,beta1=+pi*wx/vpi1)
            w0 += wy
            plt.plot(wx, wy, '.' if n<0 else '-', label=f'n = {n}')
        plt.plot(wx, w0, '-', label=f'sum')
        plt.legend(); plt.xlabel('drive voltage (V)'); plt.ylabel('$I_n$')
    mzsidebandplotwithsum(0.25, 0.75, pi/4); plt.show()
def LC2Znrf(L,C): # L in nH/cm, C in pF/cm
    Z = np.sqrt(1000*L/C)
    nrf = 299792458*np.sqrt(C*1e-12/1e-2*L*1e-9/1e-2)
    return Z,nrf
def Znrf2LC(Z,nrf):
    LC = 1e12/1e2*1e9/1e2*(nrf/299792458)**2
    LoverC = Z**2/1000
    L,C = np.sqrt(LC*LoverC),np.sqrt(LC/LoverC)
    return L,C # L in nH/cm, C in pF/cm
def cpwmglnape(g=17,h=9,buffer=0.0,loss=0.0,length=10,f0=15,term=None,air=False,λ=369,metal=4.8,gridnum=36000,gridsize=500,fmax=80):
    modeargs = dict(λ=λ,width=5,depth=0.2 if λ<1000 else 1.0,ape=16.5,rpe=2.0,apetemp=320,rpetemp=300,sell='mgln',verbose=0)
    el = CPW(g,h,{1:'MgLN',2:' ',3:'quartz',4:'hot',5:'ground'},[1,2,[5,2,4,2,5],3],[0, buffer+1e-6, buffer+metal],gridnum,gridsize=gridsize,modeargs=modeargs)
    if air:
        el = CPW(g,h,{1:'MgLN',2:' ',3:'air',4:'hot',5:'ground'},[1,2,[5,3,4,3,5],3],[0, buffer+1e-6, buffer+metal],gridnum,gridsize=gridsize,modeargs=modeargs)
    C,L,Z,Z0,nrf = [getattr(el,a) for a in ['C','L','Z','Z0','nrf']]
    # print('Z',Z,'nrf',nrf)
    term = term if term is not None else Z
    el.d = rfbandwidth(Z=Z,nrf=nrf,loss=loss,zt=term,lengthinmm=length,λ=modeargs['λ'],sell='mglnwg',fmax=fmax,plot=0)
    db = el.d['coprop'](f0) # in dB
    # print(modeargs,db,'db',Z,'Ω')
    el.vpi1 = el.vpi(xoffset=0) * 10**(abs(db)/20) * 10/length
    return el

if __name__ == '__main__':
    plots = 1
    def ktptest(mode=True): ## KTP test
        el = Electrode({1:'sub',2:'buffer',3:'KTP',4:'hot',5:'ground'},
            [3,2,[5,2,4,2,5],2,1],
            [-500, -21.5, -4.5, 4.5, 21.5, 500],
            [-500, 0, 0.4, 4.8, 54.8, 500],6000) # Vπ: 25.110volt·cm at x = 0.00µm, C=1.745pF/cm, L=4.443nH/cm, Z=50.47Ω, Z0=133.20Ω, nrf=2.639
        print(el)
        if mode:
            # ktpmodeargs = {'λ':1550,'width':4,'depth':5,'ape':0.95,'rpe':0.9,'res':0.5,'sell':'ktp','isotropic':False,'plotmode':0,'cachelookup':0,'verbose':0}
            ktpmodeargs = {'λ':1550,'width':4,'depth':5,'ape':0.95,'rpe':0.9,'res':0.5,'sell':'ktp','plotmode':0,'cachelookup':0,'verbose':0}
            el = CPW(17,9,{1:'KTP',2:'buffer',3:'sub',4:'hot',5:'ground'},[1,2,[5,2,4,2,5],2,3],[0, 0.4, 4.8, 54.8],36000,modeargs=ktpmodeargs)
            print(el)
        if plots:
            el.plotvpis()
            el.plotgamma(17)
            el.plot()
    def lntest(): ## LN test
        # el = Electrode({1:'air',2:'LN',3:'silicon',4:'hot',5:'ground'},[3,2,[5,1,4,1,5],1],[-500,-24,-4,4,24,500],[-500,-10,0,10,500],36000) # C=2.554pF/cm, L=3.983nH/cm, Z=39.49Ω, Z0=119.39Ω, nrf=3.023
        # el = Electrode({1:'air',2:'isoLN',3:'isoLN',4:'hot',5:'ground'},[3,2,[5,1,4,1,5],1],[-500,-24,-4,4,24,500],[-500,-10,0,5,500],36000) # C=3.245pF/cm, L=4.774nH/cm, Z=38.36Ω, Z0=143.12Ω, nrf=3.731
        lnmodeargs = {'λ':1550,'width':6,'depth':0.8,'ape':2.5,'rpe':0,'xcut':1,'res':0.2,'sell':'ln','plotmode':0,'cachelookup':1,'verbose':0}
        el = Electrode({1:'air',2:'LN',3:'LN',4:'hot',5:'ground'},[3,2,[5,1,4,1,5],1],[-500,-24,-4,4,24,500],[-500,-10,0,5,500],36000,modeargs=lnmodeargs) # C=3.240pF/cm, L=4.774nH/cm, Z=38.38Ω, Z0=143.12Ω, nrf=3.729
        print(el,el.minres())
        print(el.vpi())
    def gndtest():
        el = XcutElectrode(47,8,5,10,'LN',gridsize=1000,gridnum=36000)
        if plots: el.plot(pause=0,savetext='')
        if plots: el.plotmode()
        print('gndtest1',el)
        el = XcutElectrode(47,8,5,10,'LN',gnd=100,gridsize=1000,gridnum=36000)
        print('gndtest2',el)
        if plots: el.v.aty(-10).plot(m=1,pause=0)
        el = XcutElectrode(55,10,5,10,'LN',gnd=100,gridsize=1000,gridnum=36000)
        print('gndtest3',el)
    ktptest()
    lntest()
    gndtest()

    #todo: calculate half grid if symmetric
