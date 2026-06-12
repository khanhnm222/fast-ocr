import type { Theme } from 'theme-ui'

const theme: Theme = {
  colors: {
    text: '#102027',
    background: '#f4f7f5',
    primary: '#007f5f',
    secondary: '#2d6a4f',
    muted: '#5f6f72',
    highlight: '#d8f3dc',
  },
  fonts: {
    body: 'Inter, system-ui, -apple-system, Segoe UI, sans-serif',
    heading: 'Poppins, Inter, system-ui, sans-serif',
    monospace: 'Menlo, monospace',
  },
  styles: {
    root: {
      fontFamily: 'body',
      lineHeight: 1.5,
      fontWeight: 400,
    },
  },
  buttons: {
    primary: {
      cursor: 'pointer',
      borderRadius: 6,
      px: 3,
      py: 2,
      bg: 'primary',
    },
  },
}

export default theme
