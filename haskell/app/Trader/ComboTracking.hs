module Trader.ComboTracking (
    activeComboUuid,
    orderComboUuid,
) where

import Control.Applicative ((<|>))

activeComboUuid :: Maybe a -> Maybe a -> Maybe a
activeComboUuid openCombo selectedCombo =
    openCombo <|> selectedCombo

orderComboUuid :: Bool -> Maybe a -> Maybe a -> Maybe a
orderComboUuid hasOpenPosition openCombo selectedCombo =
    if hasOpenPosition
        then activeComboUuid openCombo selectedCombo
        else selectedCombo
