import numpy as np
from gcwork import starset
from gcwork.starTables import Labels
import os
import pdb

def makeLabelDat(oldLabelFile, root='./', align='align/align_t', poly='polyfit_d/fit',
                 addNewStars=True, keepOldStars=True, updateStarPosVel=True,
                 newUse=0, rad_cut=None,
                 stars=None, newLabelFile='label_new.dat'):
    """
    Make a new label.dat file using output from align and polyfit.

    Optional Inputs:
    root: The root of align analysis (e.g. './' or '07_05_18.')
    align: The root filename of the align output.
    poly: The root filename of the polyfit output.
    stars: A starset.StarSet() object with polyfit already loaded. 
           This overrides align/poly/root values and is useful for
           custom cuts that trim_align can't handle such as magnitude
           dependent velocity error cuts. BEWARE: stars may be modified.

    Outputs:
    source_list/label_new.dat

    Dependencies:
    Polyfit and align must contain the same numbers/names of stars. Also, 
    making the label.dat file depends on having the absolute astrometry
    done correctly. See gcwork.starset to learn about how the absolute
    astrometry is loaded (it depends on a specific reference epoch in align).

    You MUST run this on something that has already been run through 
    java align_absolute.
    """
    from gcwork import starset
    
    if stars == None:
        s = starset.StarSet(root + align)

        if (poly != None):
            s.loadPolyfit(root + poly)
            s.loadPolyfit(root + poly)
    else:
        s = stars

    # Trim out the new stars if we aren't going to add them
    if not addNewStars:
        idx = []
        for ss in range(len(s.stars)):
            if 'star' not in s.stars[ss].name:
                idx.append(ss)
        s.stars = [s.stars[ss] for ss in idx]

    # Get the 2D radius of all stars and sort
    radius = s.getArray('r2d')
    ridx = radius.argsort()
    s.stars = [s.stars[ss] for ss in ridx]
    

    # Get info for all the stars.
    names = np.array(s.getArray('name'))

    if poly != None:
        t0 = s.getArray('fitXv.t0')
        x = s.getArray('fitXv.p') * -1.0
        y = s.getArray('fitYv.p')
        xerr = s.getArray('fitXv.perr')
        yerr = s.getArray('fitYv.perr')
        vx = s.getArray('fitXv.v') * 995.0 * -1.0
        vy = s.getArray('fitYv.v') * 995.0
        vxerr = s.getArray('fitXv.verr') * 995.0
        vyerr = s.getArray('fitYv.verr') * 995.0
    else:
        t0 = s.getArray('fitXalign.t0')
        x = s.getArray('fitXalign.p') * -1.0
        y = s.getArray('fitYalign.p')
        xerr = s.getArray('fitXalign.perr')
        yerr = s.getArray('fitYalign.perr')
        vx = s.getArray('fitXalign.v') * 995.0 * -1.0
        vy = s.getArray('fitYalign.v') * 995.0
        vxerr = s.getArray('fitXalign.verr') * 995.0
        vyerr = s.getArray('fitYalign.verr') * 995.0

    x -= x[0]
    y -= y[0]
    r2d = np.sqrt(x**2 + y**2)
    mag = s.getArray('mag')

    # Fix Sgr A*
    idx = np.where(names == 'SgrA')[0]
    if (len(idx) > 0):
        x[idx] = 0
        y[idx] = 0
        vx[idx] = 0
        vy[idx] = 0
        vxerr[idx] = 0
        vyerr[idx] = 0
        r2d[idx] = 0

    # Clean up xerr and yerr so that they are at least 1 mas
    idx = np.where(xerr < 0.00001)[0]
    xerr[idx] = 0.00001
    idx = np.where(yerr < 0.00001)[0]
    yerr[idx] = 0.00001

    ##########
    # Load up the old star list and find the starting
    # point for new names.
    ##########
    oldLabels = Labels(labelFile=oldLabelFile)
    alnLabels = Labels(labelFile=oldLabelFile)
    newLabels = Labels(labelFile=oldLabelFile)

    if addNewStars:
        newNumber = calcNewNumbers(oldLabels.name, names)

    # Sort the old label list by radius just in case it
    # isn't already. We will update the radii first since
    # these sometimes get out of sorts.
    oldLabels.r = np.hypot(oldLabels.x, oldLabels.y)
    sidx = oldLabels.r.argsort()
    oldLabels.take(sidx)

    # Clean out the new label lists.
    newLabels.ourName = []
    newLabels.name = []
    newLabels.mag = []
    newLabels.x = []
    newLabels.y = []
    newLabels.xerr = []
    newLabels.yerr = []
    newLabels.vx = []
    newLabels.vy = []
    newLabels.vxerr = []
    newLabels.vyerr = []
    newLabels.t0 = []
    newLabels.useToAlign = []
    newLabels.r = []

    # Load up the align info into the alnLabels object
    alnLabels.ourName = names
    alnLabels.name = names
    alnLabels.mag = mag
    alnLabels.x = x
    alnLabels.y = y
    alnLabels.xerr = xerr
    alnLabels.yerr = yerr
    alnLabels.vx = vx
    alnLabels.vy = vy
    alnLabels.vxerr = vxerr
    alnLabels.vyerr = vyerr
    alnLabels.t0 = t0
    alnLabels.r = r2d

    def addStarFromAlign(alnLabels, ii, use):
        newLabels.ourName.append(alnLabels.ourName[ii])
        newLabels.name.append(alnLabels.name[ii])
        newLabels.mag.append(alnLabels.mag[ii])
        newLabels.x.append(alnLabels.x[ii])
        newLabels.y.append(alnLabels.y[ii])
        newLabels.xerr.append(alnLabels.xerr[ii])
        newLabels.yerr.append(alnLabels.yerr[ii])
        newLabels.vx.append(alnLabels.vx[ii])
        newLabels.vy.append(alnLabels.vy[ii])
        newLabels.vxerr.append(alnLabels.vxerr[ii])
        newLabels.vyerr.append(alnLabels.vyerr[ii])
        newLabels.t0.append(alnLabels.t0[ii])
        newLabels.useToAlign.append(use)
        newLabels.r.append(alnLabels.r[ii])

    def addStarFromOldLabels(oldLabels, ii):
        newLabels.ourName.append(oldLabels.name[ii])
        newLabels.ourName.append(oldLabels.ourName[ii])
        newLabels.name.append(oldLabels.name[ii])
        newLabels.mag.append(oldLabels.mag[ii])
        newLabels.x.append(oldLabels.x[ii])
        newLabels.y.append(oldLabels.y[ii])
        newLabels.xerr.append(oldLabels.xerr[ii])
        newLabels.yerr.append(oldLabels.yerr[ii])
        newLabels.vx.append(oldLabels.vx[ii])
        newLabels.vy.append(oldLabels.vy[ii])
        newLabels.vxerr.append(oldLabels.vxerr[ii])
        newLabels.vyerr.append(oldLabels.vyerr[ii])
        newLabels.t0.append(oldLabels.t0[ii])
        newLabels.useToAlign.append(oldLabels.useToAlign[ii])
        newLabels.r.append(oldLabels.r[ii])

    def deleteFromAlign(alnLabels, idx):
        # Delete them from the align lists.
        alnLabels.ourName = np.delete(alnLabels.ourName, idx)
        alnLabels.name = np.delete(alnLabels.name, idx)
        alnLabels.mag = np.delete(alnLabels.mag, idx)
        alnLabels.x = np.delete(alnLabels.x, idx)
        alnLabels.y = np.delete(alnLabels.y, idx)
        alnLabels.xerr = np.delete(alnLabels.xerr, idx)
        alnLabels.yerr = np.delete(alnLabels.yerr, idx)
        alnLabels.vx = np.delete(alnLabels.vx, idx)
        alnLabels.vy = np.delete(alnLabels.vy, idx)
        alnLabels.vxerr = np.delete(alnLabels.vxerr, idx)
        alnLabels.vyerr = np.delete(alnLabels.vyerr, idx)
        alnLabels.t0 = np.delete(alnLabels.t0, idx)
        alnLabels.r = np.delete(alnLabels.r, idx)

    # Radial cut
    if rad_cut != None:
        cut = []
        for rr in range(len(alnLabels.name)):
            if alnLabels.r[rr] > rad_cut:
                cut.append(rr)

        deleteFromAlign(alnLabels, cut)
    
    nn = 0
    while nn < len(oldLabels.name):
        #
        # First see if there are any new stars that should come
        # before this star.
        #
        if addNewStars:
            def filterFunction(i):
                return (alnLabels.r[i] < oldLabels.r[nn]) and ('star' in alnLabels.name[i])
            idx = filter(filterFunction, range(len(alnLabels.name)))

            for ii in idx:
                rAnnulus = int(math.floor(alnLabels.r[ii]))
                number = newNumber[rAnnulus]
                alnLabels.name[ii] = 'S%d-%d' % (rAnnulus, number)
                newNumber[rAnnulus] += 1

                # Insert these new stars.
                addStarFromAlign(alnLabels, ii, newUse)

            # Delete these stars from the align info.
            deleteFromAlign(alnLabels, idx)

        #
        # Now look for this star in the new align info
        #
        idx = np.where(alnLabels.name == oldLabels.name[nn])[0]
            
        if len(idx) > 0:
            # Found the star

            if updateStarPosVel:
                # Update with align info
                addStarFromAlign(alnLabels, idx[0], oldLabels.useToAlign[nn])
            else:
                # Don't update with align info
                addStarFromOldLabels(oldLabels, nn)
                
            deleteFromAlign(alnLabels, idx[0])

        elif keepOldStars:
            # Did not find the star. Only keep if user said so.
            addStarFromOldLabels(oldLabels, nn)

        nn += 1

    # Add the rest
    for gg in range(len(alnLabels.name)):
        addStarFromAlign(alnLabels, gg, newUse)
        
    # Quick verification that we don't have repeated names.
    uniqueNames = np.unique(newLabels.name)
    if len(uniqueNames) != len(newLabels.name):
        print( 'Problem, we have a repeat name!!')

    # Write to output
    newLabels.saveToFile(root + 'source_list/' + newLabelFile)


def calcNewNumbers(oldNames, newNames):
    # Loop through annuli of 1 arcsecond and find last name
    rRange = np.arange(20)
    newNumber = np.zeros(len(rRange))
    for rr in range(len(rRange)):
        substring = 'S%d-' % rr
        rNameOld = filter(lambda x: x.find(substring) != -1, oldNames)
        rNameNew = filter(lambda x: x.find(substring) != -1, newNames)
            
        if (len(rNameOld) == 0):
            newNumber[rr] = 1
        else:
            rNumberOld = np.zeros(len(rNameOld))
            for nn in range(len(rNameOld)):
                tmp = rNameOld[nn].split('-')
                rNumberOld[nn] = int(tmp[-1])
            rNumberOld.sort()
            
            if (len(rNameNew) != 0):
                rNumberNew = np.zeros(len(rNameNew))
                for nn in range(len(rNameNew)):
                    tmp = rNameNew[nn].split('-')
                    rNumberNew[nn] = int(tmp[-1])
                rNumberNew.sort()
            else:
                rNumberNew = np.array([1])
                
            newNumber[rr] = max([rNumberOld[-1], rNumberNew[-1]]) + 1

        print( 'First New Number is S%d-%d' % (rRange[rr], newNumber[rr]))

    return newNumber

def align_makeLabeldat(target, rad_cut=None, newUse=0, work_dir='./'):
    dirs = os.listdir(work_dir)
    _dirs = []
    for dd in dirs:
        if len(dd) == 37:
            _dirs.append(dd)

    for ii in _dirs:
        os.chdir(ii)
        print(ii)
        makeLabelDat('source_list/%s_label.dat' %target, newUse=newUse, rad_cut=rad_cut)
        print('\n')
        os.chdir('../')
