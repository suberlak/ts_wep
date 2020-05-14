from lsst.ts.wep.Utility import DeblendDonutType
from lsst.ts.wep.deblend.DeblendAdapt import DeblendAdapt
from lsst.ts.wep.deblend.DeblendConvolveTemplate import DeblendConvolveTemplate


class DeblendDonutFactory(object):
    """Factory for creating the deblend donut object to deblend the bright star
    donut from neighboring stars."""

    @staticmethod
    def createDeblendDonut(deblendDonutType):
        """Create the concrete deblend donut object.

        Parameters
        ----------
        deblendDonutType : enum 'DeblendDonutType'
            Algorithm to deblend the donut.

        Returns
        -------
        DeblendAdapt
            Deblend donut object.

        Raises
        ------
        ValueError
            The deblend donut type is not supported.
        """

        if (deblendDonutType == DeblendDonutType.Adapt):
            return DeblendAdapt()
        elif (deblendDonutType == DeblendDonutType.ConvolveTemplate):
            return DeblendConvolveTemplate()
        else:
            raise ValueError("The %s is not supported." % deblendDonutType)
