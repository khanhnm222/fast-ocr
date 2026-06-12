import { Button, Card, Container, Heading, Text } from 'theme-ui'

function App() {
  return (
    <Container
      sx={{
        py: 5,
      }}
    >
      <Card sx={{ p: 4, borderRadius: 8 }}>
        <Heading as="h1">Fast OCR Web</Heading>
        <Text as="p" sx={{ mt: 3, color: 'muted' }}>
          React + Vite + TypeScript + Theme UI is configured and ready.
        </Text>

        <Button
          sx={{ mt: 4 }}
          onClick={() => {
            window.alert('Theme UI is working')
          }}
        >
          Test Theme UI
        </Button>
      </Card>
    </Container>
  )
}

export default App
