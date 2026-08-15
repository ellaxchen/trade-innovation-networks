# Chord diagram using circlize — export/import split per country
# Matches migration-chord.png reference style
# Usage: Rscript chord_diagram.R [year] [top_n]
library(circlize)
library(data.table)

args <- commandArgs(trailingOnly = TRUE)
YEAR <- if (length(args) >= 1) as.integer(args[1]) else 2015
TOP_N <- if (length(args) >= 2) as.integer(args[2]) else 4

# ---- Load trade data ----
trade <- fread("../processed/trade_edges_2015.csv")

# Filter year and aggregate by country pair
trade <- trade[year == YEAR]
trade <- trade[, .(value = sum(trade_value)), by = .(exporter, importer)]

# ---- Load citation data ----
cite <- fread("../processed/country_citation_edges_by_year.csv")
setnames(cite, c("citing_country", "cited_country", "year", "weight"),
         c("exporter", "importer", "year", "value"))
cite <- cite[year == YEAR]
# Filter out non-country offices
non_country <- c("EP", "WO", "DD", "OA", "AP", "GC", "IB", "EM", "BX", "YU", "CS", "SU")
cite <- cite[!(exporter %in% non_country) & !(importer %in% non_country)]
cite <- cite[exporter != importer]

# ---- Map country codes ----
codes <- fread("../rawdata/country_codes_V202601.csv")
baci_to_iso <- setNames(codes$country_iso2, codes$country_code)
baci_to_iso["490"] <- "TW"
name_to_iso <- setNames(codes$country_iso2, codes$country_name)
name_to_iso["Other Asia, nes"] <- "TW"
iso_to_name <- setNames(codes$country_name, codes$country_iso2)
iso_to_name["TW"] <- "Taiwan"

# ---- Top N countries by total flow ----
trade[, `:=`(source = baci_to_iso[as.character(exporter)],
             target = baci_to_iso[as.character(importer)])]
trade <- trade[!is.na(source) & !is.na(target) & source != target]
trade <- trade[, .(value = sum(value)), by = .(source, target)]

cite <- cite[!is.na(exporter) & !is.na(importer)]

trade_flows <- rbind(
  trade[, .(country = source, flow = value)],
  trade[, .(country = target, flow = value)]
)[, .(total = sum(flow)), by = country]

cite_flows <- rbind(
  cite[, .(country = exporter, flow = value)],
  cite[, .(country = importer, flow = value)]
)[, .(total = sum(flow)), by = country]

all_flows <- merge(trade_flows, cite_flows, by = "country", all = TRUE, suffixes = c("_trade", "_cite"))
all_flows[is.na(all_flows)] <- 0
all_flows[, total := total_trade + total_cite]
top_countries <- all_flows[order(-total)][1:TOP_N]$country
cat(sprintf("Top %d: %s\n", TOP_N, paste(top_countries, collapse = ", ")))

# ---- Colors ----
colors <- c("#E63946", "#3A7CA5", "#2A9D8F", "#F4A261", "#9B5DE5", "#E76F51",
            "#264653", "#A8DADC", "#E9C46A", "#6D597A", "#B5838D", "#4895EF",
            "#FF6B6B", "#4ECDC4", "#FFD166", "#F15BB5", "#00BBF9", "#FEE440",
            "#D4A373", "#073B4C", "#888888")
names(colors) <- top_countries

# ---- Build export->import matrix ----
# Rows: exports (source), Columns: imports (target)
# Each country has sector_Xe (export) and sector_Xi (import)

build_matrix <- function(df, val_col, countries) {
  # Create row and column names
  row_names <- paste0(countries, "_e")  # export sectors
  col_names <- paste0(countries, "_i")  # import sectors
  all_names <- c(row_names, col_names)

  # Empty square matrix
  mat <- matrix(0, nrow = length(all_names), ncol = length(all_names))
  rownames(mat) <- all_names
  colnames(mat) <- all_names

  # Fill: [source_e, target_i] = flow value
  for (i in seq_len(nrow(df))) {
    src <- df[[i, "source"]]
    tgt <- df[[i, "target"]]
    val <- df[[i, ..val_col]][[1]]  # data.table column access
    if (src %in% countries && tgt %in% countries && src != tgt && val > 0) {
      row_idx <- paste0(src, "_e")
      col_idx <- paste0(tgt, "_i")
      mat[row_idx, col_idx] <- val
    }
  }
  return(mat)
}

trade_mat <- build_matrix(trade, "value", top_countries)
cite_mat <- build_matrix(cite, "value", top_countries)

# ---- Group: export and import sectors share the same country group ----
all_sectors <- rownames(trade_mat)
group <- setNames(gsub("_[ei]$", "", all_sectors), all_sectors)
# Order groups: match top_countries order
group <- factor(group, levels = top_countries)

# ---- Grid colors: export = solid color, import = faded (lighter) ----
grid_cols <- c()
for (cn in top_countries) {
  col_rgb <- col2rgb(colors[cn]) / 255
  # Export: full color
  grid_cols[paste0(cn, "_e")] <- colors[cn]
  # Import: faded (mix with white)
  faded <- rgb(col_rgb[1] * 0.6 + 0.4, col_rgb[2] * 0.6 + 0.4, col_rgb[3] * 0.6 + 0.4)
  grid_cols[paste0(cn, "_i")] <- faded
}

# Sector order: alternate export/import per country
sector_order <- c(rbind(paste0(top_countries, "_e"), paste0(top_countries, "_i")))

# ---- Draw function ----
draw_chord <- function(mat, title_str, filename, ...) {
  png(filename, width = 1400, height = 1400, res = 150)
  circos.clear()
  circos.par(start.degree = 90, gap.after = c(rep(c(2, 8), length(top_countries))))
  # gap.after: 2° after export, 8° after import (creates country grouping)
  # But total gaps must sum to 360. Adjust: each country gets 2+8=10°, N countries = N*10°
  # Remaining = 360 - N*10 for sectors
  # Actually the gap.after controls gaps between consecutive sectors.
  # With order: CN_e, CN_i, US_e, US_i, DE_e, DE_i, JP_e, JP_i
  # gap[1]=2 after CN_e (small gap before CN_i)
  # gap[2]=8 after CN_i (big gap before US_e)
  # gap[3]=2 after US_e (small gap before US_i)
  # gap[4]=8 after US_i (big gap before DE_e)
  # etc.

  chordDiagram(mat,
    group = group,
    grid.col = grid_cols,
    directional = 1,           # rows -> columns (export -> import)
    direction.type = c("diffHeight", "arrows"),
    diffHeight = mm_h(2),
    link.arr.type = "big.arrow",
    annotationTrack = c("grid", "axis"),
    preAllocateTracks = list(
      track.height = mm_h(5),
      track.margin = c(mm_h(6), 0)
    ),
    order = sector_order,
    scale = TRUE,
    ...)

  # Add country labels
  circos.track(track.index = 2, panel.fun = function(x, y) {
    sector.index <- get.cell.meta.data("sector.index")
    xlim <- get.cell.meta.data("xlim")
    ylim <- get.cell.meta.data("ylim")
    country <- gsub("_[ei]$", "", sector.index)
    side <- ifelse(grepl("_e$", sector.index), "Export", "Import")
    circos.text(mean(xlim), mean(ylim), country, cex = 0.7,
                niceFacing = TRUE, facing = "clockwise")
  }, bg.border = NA)

  title(title_str, cex.main = 1.2)
  circos.clear()
  dev.off()
  cat(sprintf("Saved -> %s\n", filename))
}

# ---- Generate plots ----
draw_chord(trade_mat,
  sprintf("Trade Flows (%d)", YEAR),
  sprintf("../plots/chord_trade_%d.png", YEAR))

draw_chord(cite_mat,
  sprintf("Patent Citations (%d)", YEAR),
  sprintf("../plots/chord_cite_%d.png", YEAR))

cat("Done.\n")
