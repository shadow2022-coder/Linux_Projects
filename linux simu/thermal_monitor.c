

#include "raylib.h"
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── constants ──────────────────────────────────────────────────────────── */
#define MAX_ZONES      64
#define MAX_STR        128
#define WIN_W          900
#define WIN_H          640
#define HEADER_H       95
#define ROW_H          74
#define BAR_MAX_TEMP  100.0f
#define FPS            5

/* ── data ───────────────────────────────────────────────────────────────── */
typedef struct {
    char  name[MAX_STR];   /* e.g. thermal_zone0          */
    char  type[MAX_STR];   /* e.g. x86_pkg_temp, acpitz … */
    float temp;            /* Celsius                      */
} Zone;

static Zone zones[MAX_ZONES];
static int  zone_count = 0;

/* ── read thermal data from sysfs ───────────────────────────────────────── */
static void read_thermal_data(void)
{
    zone_count = 0;
    char path[256], buf[64];
    FILE *f;

    for (int i = 0; i < MAX_ZONES; i++) {
        /* temperature */
        snprintf(path, sizeof(path),
                 "/sys/class/thermal/thermal_zone%d/temp", i);
        f = fopen(path, "r");
        if (!f) break;

        buf[0] = '\0';
        if (!fgets(buf, sizeof(buf), f)) { fclose(f); break; }
        fclose(f);
        zones[zone_count].temp = (float)atoi(buf) / 1000.0f;

        /* zone name */
        snprintf(zones[zone_count].name, MAX_STR, "thermal_zone%d", i);

        /* type  (acpitz / x86_pkg_temp / INT3400 / …) */
        snprintf(path, sizeof(path),
                 "/sys/class/thermal/thermal_zone%d/type", i);
        f = fopen(path, "r");
        if (f) {
            if (fgets(zones[zone_count].type, MAX_STR, f))
                zones[zone_count].type[
                    strcspn(zones[zone_count].type, "\n\r")] = '\0';
            fclose(f);
        } else {
            strcpy(zones[zone_count].type, "unknown");
        }

        zone_count++;
    }
}

/* colour ramp: green → yellow → orange → red */
static Color temp_color(float t)
{
    if (t < 40.0f) return (Color){ 40, 200,  80, 255 };
    if (t < 60.0f) return (Color){220, 200,  30, 255 };
    if (t < 75.0f) return (Color){240, 130,  20, 255 };
    return                (Color){220,  40,  40, 255 };
}

/* filled rounded bar with border */
static void draw_bar(int x, int y, int w, int h, float fraction, Color col)
{
    if (fraction < 0.0f) fraction = 0.0f;
    if (fraction > 1.0f) fraction = 1.0f;

    DrawRectangleRounded(
        (Rectangle){(float)x, (float)y, (float)w, (float)h},
        0.45f, 8, (Color){22, 22, 22, 255});

    int fw = (int)(w * fraction);
    if (fw > 6)
        DrawRectangleRounded(
            (Rectangle){(float)x, (float)y, (float)fw, (float)h},
            0.45f, 8, col);

    DrawRectangleRoundedLines(
        (Rectangle){(float)x, (float)y, (float)w, (float)h},
        0.45f, 8, (Color){110, 110, 110, 200});
}

/* ── entry point ─────────────────────────────────────────────────────────── */
int main(void)
{
    InitWindow(WIN_W, WIN_H, "Thermal Monitor  [raylib + raygui]");
    SetTargetFPS(FPS);

    

    float scroll    = 0.0f;
    float anim_time = 0.0f;

    /* raygui visual style */
    GuiSetStyle(DEFAULT, TEXT_SIZE,             15);
    GuiSetStyle(BUTTON,  BASE_COLOR_NORMAL,     ColorToInt((Color){ 55, 55, 60, 255}));
    GuiSetStyle(BUTTON,  BASE_COLOR_FOCUSED,    ColorToInt((Color){ 75, 75, 85, 255}));
    GuiSetStyle(BUTTON,  BASE_COLOR_PRESSED,    ColorToInt((Color){ 40, 40, 45, 255}));
    GuiSetStyle(BUTTON,  BORDER_COLOR_NORMAL,   ColorToInt((Color){110,110,120, 255}));
    GuiSetStyle(BUTTON,  TEXT_COLOR_NORMAL,     ColorToInt(RAYWHITE));

    while (!WindowShouldClose()) {
        anim_time += GetFrameTime();

        /* ── scrolling ───────────────────────────────────────────────── */
        scroll -= GetMouseWheelMove() * 42.0f;
        float content_h  = (float)(zone_count * ROW_H);
        float view_h     = (float)(WIN_H - HEADER_H - 28);
        float max_scroll = content_h > view_h ? content_h - view_h : 0.0f;
        if (scroll < 0.0f)        scroll = 0.0f;
        if (scroll > max_scroll)  scroll = max_scroll;
        
        read_thermal_data();
        /* ── rendering ───────────────────────────────────────────────── */
        BeginDrawing();
        ClearBackground((Color){24, 24, 26, 255});

        /* ── header bar ──────────────────────────────────────────────── */
        DrawRectangle(0, 0, WIN_W, HEADER_H, (Color){33, 33, 36, 255});
        DrawText("  Thermal Monitor", 12, 14, 28, RAYWHITE);
        DrawText("  Live sensor data  \xe2\x80\x94  /sys/class/thermal/",
                 14, 50, 14, (Color){140, 140, 150, 255});

        if (GuiButton((Rectangle){WIN_W - 148.0f, 22.0f, 128.0f, 38.0f},
                      "#211# Refresh")) {
            read_thermal_data();
            scroll = 0.0f;
        }

        /* ── column header row ───────────────────────────────────────── */
        int chy = HEADER_H;
        DrawRectangle(0, chy, WIN_W, 26, (Color){42, 42, 46, 255});
        DrawLine(0, chy, WIN_W, chy, (Color){60,60,65,255});
        DrawText("#",           10,  chy + 6, 13, (Color){130,130,140,255});
        DrawText("Zone",        42,  chy + 6, 13, (Color){130,130,140,255});
        DrawText("Type",       210,  chy + 6, 13, (Color){130,130,140,255});
        DrawText("Temp",       400,  chy + 6, 13, (Color){130,130,140,255});
        DrawText("Visualizer", 510,  chy + 6, 13, (Color){130,130,140,255});

        /* ── zone rows ───────────────────────────────────────────────── */
        int clip_y = HEADER_H + 26;
        int clip_h = WIN_H - clip_y - 28;
        BeginScissorMode(0, clip_y, WIN_W, clip_h);

        for (int i = 0; i < zone_count; i++) {
            Zone *z  = &zones[i];
            int   ry = clip_y + i * ROW_H - (int)scroll;

            if (ry + ROW_H < clip_y || ry > WIN_H) continue;

            /* alternating row background */
            DrawRectangle(0, ry, WIN_W, ROW_H - 2,
                (i & 1) ? (Color){33,33,36,255} : (Color){40,40,44,255});

            Color tc = temp_color(z->temp);

            /* pulsing left accent for hot sensors (≥ 75 °C) */
            Color accent = tc;
            if (z->temp >= 75.0f)
                accent.a = (unsigned char)(170 + sinf(anim_time * 5.0f) * 85.0f);
            DrawRectangle(0, ry, 4, ROW_H - 2, accent);

            /* index */
            char idx[6];
            snprintf(idx, sizeof(idx), "%d", i + 1);
            DrawText(idx, 10, ry + 28, 14, (Color){130,130,140,255});

            /* zone name */
            DrawText(z->name,  42, ry + 14, 14, (Color){200,200,210,255});

            /* type  (coloured) */
            DrawText(z->type, 210, ry + 14, 15, (Color){90, 175, 255, 255});

            /* ── temperature — large text ─────────────────────────────── */
            char ts[16];
            snprintf(ts, sizeof(ts), "%.1f", z->temp);
            DrawText(ts,  400, ry + 10, 28, tc);
            int tw = MeasureText(ts, 28);
            DrawText("\xc2\xb0""C", 400 + tw + 3, ry + 16, 15,
                     (Color){170,170,180,255});

            /* ── bar visualizer ──────────────────────────────────────── */
            float frac = z->temp / BAR_MAX_TEMP;
            int bx = 510, by = ry + 20, bw = 350, bh = 28;
            draw_bar(bx, by, bw, bh, frac, tc);

            /* percentage label centred on bar */
            char pct[10];
            snprintf(pct, sizeof(pct), "%.0f%%", frac * 100.0f);
            int pw = MeasureText(pct, 13);
            DrawText(pct, bx + bw / 2 - pw / 2, by + 7, 13, RAYWHITE);

            /* tick marks at 40 / 60 / 80 °C */
            int ticks[] = {40, 60, 80};
            for (int t = 0; t < 3; t++) {
                int tx = bx + (int)(bw * ticks[t] / (int)BAR_MAX_TEMP);
                DrawLine(tx, by + bh + 1, tx, by + bh + 5,
                         (Color){110,110,110,200});
                char tl[4];
                snprintf(tl, sizeof(tl), "%d", ticks[t]);
                DrawText(tl, tx - 7, by + bh + 6, 10, (Color){100,100,110,200});
            }
        }

        EndScissorMode();

        /* ── footer ──────────────────────────────────────────────────── */
        DrawRectangle(0, WIN_H - 28, WIN_W, 28, (Color){33,33,36,255});
        DrawLine(0, WIN_H - 28, WIN_W, WIN_H - 28, (Color){55,55,60,255});
        char footer[128];
        snprintf(footer, sizeof(footer),
                 "  %d zones found   |   scroll: mouse wheel   "
                 "|   bar scale: 0 \xe2\x80\x93 100 \xc2\xb0""C",
                 zone_count);
        DrawText(footer, 4, WIN_H - 20, 12, (Color){100,100,110,255});

        EndDrawing();
    }

    CloseWindow();
    return 0;
}