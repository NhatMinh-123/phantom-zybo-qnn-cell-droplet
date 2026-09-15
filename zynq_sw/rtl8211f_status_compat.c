/* Vitis 2022.2 expects RTL8211E status layout; revision F uses page A43.
 * Register definitions: Digilent/u-boot-digilent, drivers/net/phy/realtek.c.
 * Scope the translation to F silicon and default-page status reads only.
 */
#include "xemacps.h"
#include "xil_printf.h"

int __real_XEmacPs_PhyRead(XEmacPs *, u32, u32, u16 *);

int __wrap_XEmacPs_PhyRead(XEmacPs *mac, u32 phy, u32 reg, u16 *value) {
    u16 id1, id2, page, status;
    int result;
    static int announced;
    if (reg != 0x11U) return __real_XEmacPs_PhyRead(mac, phy, reg, value);
    if (__real_XEmacPs_PhyRead(mac, phy, 2U, &id1) != XST_SUCCESS ||
        __real_XEmacPs_PhyRead(mac, phy, 3U, &id2) != XST_SUCCESS)
        return XST_FAILURE;
    if (!announced) xil_printf("PHY addr=%d ID=%04x:%04x\r\n", phy, id1, id2);
    if (id1 != 0x001CU || id2 != 0xC916U)
        return __real_XEmacPs_PhyRead(mac, phy, reg, value);
    if (__real_XEmacPs_PhyRead(mac, phy, 31U, &page) != XST_SUCCESS)
        return XST_FAILURE;
    if (!announced) xil_printf("PHY page=%04x\r\n", page);
    if (page != 0U && page != 0xA42U)
        return __real_XEmacPs_PhyRead(mac, phy, reg, value);
    if (XEmacPs_PhyWrite(mac, phy, 31U, 0xA43U) != XST_SUCCESS)
        return XST_FAILURE;
    result = __real_XEmacPs_PhyRead(mac, phy, 0x1AU, &status);
    if (XEmacPs_PhyWrite(mac, phy, 31U, page) != XST_SUCCESS)
        return XST_FAILURE;
    if (result != XST_SUCCESS) return result;
    *value = (u16)(((status & 0x30U) << 10) |
                  ((status & 4U) << 8) | ((status & 8U) << 10));
    if (!announced) {
        xil_printf("RTL8211F detected addr=%d status=%04x\r\n", phy, status);
        announced = 1;
    }
    return XST_SUCCESS;
}
