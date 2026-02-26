// Declare WGSL files imported with Vite's ?raw query as plain strings
declare module "*.wgsl?raw" {
  const content: string;
  export default content;
}
